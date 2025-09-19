"""
Find limits with lower bound
"""

import os

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import click
import dask
import numpy as np
import xarray as xr
from dask.graph_manipulation import bind

from mmdisk.cli_utils import (
    cleanup_limits,
    dask_cluster_options,
    get_continuous_chunks,
    start_dask_cluster_and_client,
    wait_for_futures,
)
from mmdisk.fea.common import default_cycle
from mmdisk.fea.fenics.lower_bound import lower_bound_fea
from mmdisk.fea.fenics.mesh import mesh_and_properties


def find_limit_for_disk(disks, nn=19, tmp_folder=None):
    xr_ds = xr.Dataset(
        {
            "limits": xr.DataArray(
                np.full((len(disks.disks), 4, nn), np.nan),
                dims=["disks", "variables", "nn"],
                coords={
                    "variables": [
                        "T_EL_mean",
                        "Omega_EL",
                        "T_SD_mean",
                        "Omega_SD",
                    ],
                    "nn": np.arange(nn),
                    "disks": disks.disks,
                },
            )
        }
    )
    xr_ds["limits"].encoding = {"fill_value": np.nan}

    if disks.sizes["disks"] == 0:
        return xr_ds

    for i_disk, n in enumerate(disks.disks):
        disk = disks.sel(disks=n)

        if tmp_folder is not None:
            tmp_file = os.path.join(tmp_folder, f"disk_{n.item()}.npy")
            if os.path.exists(tmp_file):
                limits = np.load(tmp_file)
                xr_ds.limits.values[i_disk, :, : limits.shape[-1]] = limits
                continue

        if "heat_time" in disk:
            heat_time = disk.heat_time.item()
        else:
            heat_time = 20.0

        l_cycle = default_cycle.copy()
        l_cycle[1] = heat_time  # [0]

        mesh, props = mesh_and_properties(disk)

        limits = lower_bound_fea(mesh, props, l_cycle, num_dir=nn)

        limits = np.array(limits)[None, ...]
        if tmp_folder is not None:
            np.save(tmp_file, limits)
        xr_ds.limits.values[i_disk, :, : limits.shape[-1]] = limits
    return xr_ds


@click.command()
@click.option(
    "--case", "-c", type=click.Path(exists=True), help="Input Zarr DB", required=True
)
@click.option(
    "--output", "-o", type=click.Path(), help="Output database", required=True
)
@click.option(
    "--tmp-folder",
    "-t",
    type=click.Path(),
    help="Temporary folder for intermediate results",
)
@click.option("--nn", default=19)
@click.option(
    "--compute-chunk",
    default=10,
    help="Size of the disk chunks to group for computation",
)
@click.option("--chunksize", default=100, help="Chunk size for storage.")
@dask_cluster_options
def find_limits(
    case,
    output,
    tmp_folder,
    nn,
    compute_chunk,
    chunksize,
    queue,
    n_cores,
    n_jobs,
    time_limit,
    network,
    network_sched,
):
    """Find elastic and shakedown limits for disks in a Zarr database using lower bound FEA."""
    if compute_chunk > chunksize:
        raise ValueError(
            f"compute_chunk ({compute_chunk}) must be less than or equal to chunksize ({chunksize})."
        )

    if (chunksize % compute_chunk) != 0:
        raise ValueError(
            f"chunksize ({chunksize}) must be a multiple of compute_chunk ({compute_chunk})."
        )

    client = start_dask_cluster_and_client(
        queue,
        n_cores,
        n_jobs,
        time_limit,
        f"{n_cores * 10}GB",
        f"{n_cores * 5}GB",
        network,
        network_sched,
        ["export OMP_NUM_THREADS=1"],
    )
    print(client)
    print(client.dashboard_link)

    datavars_of_interest = ["properties", "heat_time", "geometry", "upper_density"]

    # Open to inspect metadata
    ds = xr.open_zarr(case)

    # Compute the chunk size for disks
    c_chunk_def = {k: -1 for k in ds.dims if k != "disks"}
    c_chunk_def["disks"] = compute_chunk

    # Ensure the dataset has the required data variables
    to_drop = [dv for dv in ds.data_vars if dv not in datavars_of_interest]

    # Reopen with the specified chunk size and drop unnecessary variables
    chunked_ds = xr.open_zarr(case, chunks=c_chunk_def, drop_variables=to_drop)

    # Prepare temporary folder if specified
    if tmp_folder is not None and not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # Prepare output directory
    if not os.path.exists(output):
        # Create the output file metadata
        output_ds = chunked_ds.map_blocks(
            find_limit_for_disk, kwargs=dict(nn=nn, tmp_folder=tmp_folder)
        )
        s_chunk_def = {k: -1 for k in output_ds.dims if k != "disks"}
        s_chunk_def["disks"] = chunksize
        output_ds.chunk(s_chunk_def).to_zarr(
            output, compute=False, encoding={"limits": {"fill_value": np.nan}}
        )
        missing_disks = ds.disks
    else:
        # Open the output file metadata
        output_ds = xr.open_zarr(output)
        # Check storage chunk size
        if output_ds.chunksizes["disks"][0] != chunksize:
            raise ValueError(
                f"Output dataset has a different chunk size for disks ({output_ds.chunksizes['disks'][0]}) than desired ({chunksize})."
            )
        s_chunk_def = {k: -1 for k in output_ds.dims if k != "disks"}
        s_chunk_def["disks"] = chunksize

        # Check if we need to adjust metadata
        if output_ds.sizes["disks"] < ds.sizes["disks"]:
            # If the output file is smaller, we need to adjust the metadata
            chunked_ds.sel(disks=slice(output_ds.sizes["disks"], None)).map_blocks(
                find_limit_for_disk, kwargs=dict(nn=nn, tmp_folder=tmp_folder)
            ).chunk(s_chunk_def).to_zarr(
                output,
                compute=False,
                mode="a",
                append_dim="disks",
            )
            output_ds = xr.open_zarr(output)
        # Find disks that are missing limits
        missing_disks = output_ds.disks[
            output_ds["limits"].isnull().all(["nn", "variables"]).compute()
        ]

    max_tasks = n_cores * n_jobs

    zarr_args = {"mode": "r+", "region": "auto"}
    futures = []
    prepared = 0
    batches = get_continuous_chunks(missing_disks.values, chunksize * (max_tasks // 2))
    print(f"There are {len(batches)} batches of disks.")

    for contiguous in batches:
        ds_slice = chunked_ds.sel(disks=contiguous)

        prior = ds_slice.map_blocks(
            find_limit_for_disk,
            kwargs=dict(nn=nn, tmp_folder=tmp_folder),
        ).chunk(s_chunk_def)

        storage_task = prior.to_zarr(output, compute=False, **zarr_args)
        # Bind the storage task to the client
        cleanup_task = bind(
            cleanup_limits(prior.disks, tmp_folder=tmp_folder), storage_task
        ).compute(sync=False)
        cleanup_task.num_simulations = ds_slice.sizes["disks"]

        futures.append(cleanup_task)

        prepared += ds_slice.sizes["disks"]
        if prepared >= max_tasks * chunksize * 2:
            print(f"Prepared {prepared} disks.")
            prepared -= wait_for_futures(futures)

    wait_for_futures(futures)

    client.close()


if __name__ == "__main__":
    find_limits()
