import gc
import os

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import click
import dask.array as da
import numpy as np
import xarray as xr
from mpi4py import MPI
from petsc4py import PETSc
from dask.graph_manipulation import bind

from mmdisk.cli_utils import (
    cleanup_limits,
    dask_cluster_options,
    get_continuous_chunks,
    start_dask_cluster_and_client,
    wait_for_futures,
)
from mmdisk.fea.common import create_time_points, default_cycle, default_n_cycles
from mmdisk.fea.fenics.incremental import ConvergenceError, incremental_cyclic_fea
from mmdisk.fea.fenics.mesh import mesh_and_properties, mesh_max_size
from mmdisk.fea.fenics.utils import (
    get_loadfunc,
    get_time_list_alt,
)


_dim_mapping = {
    "u": ["disks", "op", "cycles", "fea_nodes", "coordinates"],
    "sig": ["disks", "op", "cycles", "fea_cells", "stress_components"],
    "T": ["disks", "op", "cycles", "fea_nodes"],
    "PEEQ": ["disks", "op", "cycles", "fea_cells"],
    "strain": ["disks", "op", "cycles", "fea_cells", "strain_components"],
}


def _dims_and_coords(var, dataset, cycles, max_dim_cells, max_dim_nodes):
    """Return the dimensions and coordinates for the given variable."""
    dims = _dim_mapping[var]
    coords = {"disks": dataset.disks, "op": dataset.op, "cycles": cycles}
    if "fea_nodes" in dims:
        coords["fea_nodes"] = np.arange(max_dim_nodes)
    if "fea_cells" in dims:
        coords["fea_cells"] = np.arange(max_dim_cells)
    if "stress_components" in dims:
        coords["stress_components"] = ["S11", "S22", "S33", "S12"]
    if "coordinates" in dims:
        coords["coordinates"] = ["x", "z"]
    if "strain_components" in dims:
        coords["strain_components"] = ["eps_rr", "eps_theta", "eps_zz", "eps_rz"]

    shape = tuple(len(coords[dim]) if dim in coords else 0 for dim in dims)

    return dims, coords, shape


def simulate_single_case(
    disk_op,
    displacement_bc="fix-free",
    plastic_inverval=1,
    n_cycles=default_n_cycles,
    time_division=None,
    output_division=0.25,
    outputs=None,
):
    mesh, props = mesh_and_properties(disk_op)
    omega = disk_op.operations.sel(parameters="omega").item()
    T_load = disk_op.operations.sel(parameters="delta_T").item()
    t_heatdwell = disk_op.operations.sel(parameters="heat_time").item()
    t_cooldwell = t_heatdwell * 20  # cooling dwell time

    l_cycle = default_cycle.copy()
    l_cycle[1] = t_heatdwell
    l_cycle[3] = t_cooldwell

    if time_division is None:
        time_division = [0.25, 0.1, 0.25, 0.01]

    time_steps = l_cycle.copy() * time_division

    load = get_loadfunc(T_load, l_cycle)

    # t_fine = l_cycle[:3].sum() + l_cycle[2]
    period, _, t_output = create_time_points(
        n_cycles, l_cycle, output_division=output_division, skip_cold=True
    )
    t_list = get_time_list_alt(l_cycle, time_steps, n_cycles, skip_cold=True)

    return incremental_cyclic_fea(
        mesh,
        props,
        load,
        omega,
        t_list,
        t_output,
        period,
        Displacement_BC=displacement_bc,
        plastic_inverval=plastic_inverval,
        skip_cold=True,
        outputs=outputs,
    )


def simulate(
    dataset,
    n_cycles=default_n_cycles,
    displacement_bc="fix-free",
    output_division=0.25,
    output_variables=None,
    max_dim_cells=None,
    max_dim_nodes=None,
    tmp_folder=None,
):
    """Simulate the chunk of dataset of disk compositions using FEA at the
    specified operating conditions."""
    # Clean up memory
    gc.collect()
    PETSc.garbage_cleanup(MPI.COMM_SELF)

    # Prepare variables
    pg, _, t_global = create_time_points(
        n_cycles, np.ones_like(default_cycle), output_division=output_division
    )
    t_global /= pg

    if output_variables is None:
        output_variables = ["PEEQ"]

    xr_ds = xr.Dataset(
        {
            "errors": xr.DataArray(
                np.full((len(dataset.disks), len(dataset.op)), -1, np.int8),
                dims=["disks", "op"],
                coords={
                    "disks": dataset.disks,
                    "op": dataset.op,
                },
                name="errors",
            ),
        }
    )
    xr_ds["errors"].encoding = {"fill_value": -1}

    for var in output_variables:
        dims, coords, shape = _dims_and_coords(
            var, dataset, t_global, max_dim_cells, max_dim_nodes
        )
        xr_ds[var] = xr.DataArray(
            np.full(shape, np.nan, dtype=np.float32), dims=dims, coords=coords
        )
        xr_ds[var].encoding = {"fill_value": np.nan}

    # Process each disk and OP individually
    for i_d, d in enumerate(dataset.disks):
        for i_op, op in enumerate(dataset.op):
            if tmp_folder is not None:
                tmp_file = os.path.join(tmp_folder, f"disk_{d.item()}_{op.item()}.nc")
                if os.path.exists(tmp_file):
                    # Load existing results if available
                    xr_ds_i = xr.open_dataset(tmp_file, engine="netcdf4")
                    for var in output_variables:
                        xr_ds[var].values[i_d, i_op] = xr_ds_i[var].values
                    xr_ds.errors[i_d, i_op] = xr_ds_i.errors.item()
                    continue
            disk = dataset.sel(disks=d, op=op)
            try:
                _, _, flag = simulate_single_case(
                    disk,
                    displacement_bc,
                    n_cycles=n_cycles,
                    output_division=output_division,
                    outputs={
                        k: xr_ds[k][i_d, i_op].to_numpy() for k in output_variables
                    },
                )
                xr_ds.errors[i_d, i_op] = flag
            except ConvergenceError:
                # peeq = e.peeq
                xr_ds.errors[i_d, i_op] = -2
            except RuntimeError:
                xr_ds.errors[i_d, i_op] = -3

            # Store data if tmp_folder is specified
            if tmp_folder is not None:
                xr_ds.sel(disks=d, op=op).to_netcdf(tmp_file)

    return xr_ds


@click.command()
@click.option(
    "--disks", "-d", type=click.Path(exists=True), help="Input Zarr DB with disks"
)
@click.option(
    "--cases",
    "-c",
    type=click.Path(exists=True),
    help="Simulation cases and output database",
)
@click.option(
    "--tmp-folder",
    "-t",
    type=click.Path(),
    help="Temporary folder for intermediate results",
)
@click.option(
    "--data-vars",
    default=["PEEQ"],
    type=str,
    help="Data variables to store",
    multiple=True,
)
@click.option("--output-division", default=0.25, type=float)
@click.option("--n-cycles", default=default_n_cycles, type=int)
@dask_cluster_options
def main(
    disks,
    cases,
    tmp_folder,
    data_vars,
    n_cycles,
    output_division,
    queue,
    n_cores,
    n_jobs,
    time_limit,
    network,
    network_sched,
):
    client = start_dask_cluster_and_client(
        queue,
        n_cores,
        n_jobs,
        time_limit,
        f"{n_cores * 20}GB",
        f"{n_cores * 10}GB",
        network,
        network_sched,
        ["export OMP_NUM_THREADS=1"],
    )
    print(client)
    print(client.dashboard_link)

    # Prepare temporary folder if specified
    if tmp_folder is not None and not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # Load disks
    datavars_of_interest = ["properties", "heat_time", "geometry", "upper_density"]

    disks = xr.open_zarr(disks)
    doi = [v for v in disks.data_vars if v in datavars_of_interest]
    disks = disks[doi]

    chunks_def = {k: -1 for k in disks.dims if k != "disks"}
    chunks_def["disks"] = 5  # Chunk disks for parallel processing
    chunks_def["op"] = 5
    chunksize = chunks_def["disks"] * chunks_def["op"]

    # Prepare simulation arguments
    # Load mesh
    disk = disks.isel(disks=0).load()
    max_dim_nodes, max_dim_cells = mesh_max_size(disk)

    simulation_args = {
        "n_cycles": n_cycles,
        "displacement_bc": "fix-free",
        "output_division": output_division,
        "output_variables": data_vars,
        "max_dim_cells": max_dim_cells,
        "max_dim_nodes": max_dim_nodes,
        "tmp_folder": tmp_folder,
    }

    # Load cases
    simulations = xr.open_zarr(cases)
    disks_ops = xr.merge([disks, simulations[["operations"]]], join="inner").chunk(
        chunks_def
    )

    # Check if the simulations dataset has already been computed
    if "PEEQ" not in simulations.data_vars:
        # Prepare simulation dataset with metadata
        (
            disks_ops.map_blocks(simulate, kwargs=simulation_args).to_zarr(
                cases, compute=False, mode="a"
            )
        )
        missing_disks = disks_ops.disks
    else:
        # Some results have already been computed
        empty_peeq = (
            (simulations.errors == -1).any("op")
            | (simulations.PEEQ.isnull().all(["fea_cells", "cycles"])).any("op")
        ).compute()
        missing_disks = disks_ops.disks[empty_peeq]

    store_args = {"mode": "r+", "region": "auto"}
    max_tasks = n_cores * n_jobs
    futures = []
    prepared = 0
    batches = get_continuous_chunks(missing_disks.values, chunksize * (max_tasks // 2))
    print(f"There are {len(batches)} batches of disks.")

    for contiguous in batches:
        ds_slice = disks_ops.sel(disks=contiguous)
        prior = ds_slice.map_blocks(simulate, kwargs=simulation_args)
        storage_task = prior.to_zarr(cases, compute=False, **store_args)
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
    main()
