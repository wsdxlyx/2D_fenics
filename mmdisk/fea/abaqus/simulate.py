import os
import pickle
import sys

import click
import inpRW
import numpy as np
import xarray as xr
from dask.distributed import Client, Reschedule
from dask_jobqueue import SLURMCluster

from mmdisk.cli_utils import dask_cluster_options, slurm_time_to_minutes_minus_five
from mmdisk.materials.utils import approximate_cooling_time
from mmdisk.utils import dimensional_geometry, pad_arrays

from ..common import default_cycle, default_n_cycles, create_time_points
from .prepare import (
    convert_to_elastic,
    parse_mesh_from_input_file,
    rename,
    scale_mesh,
    set_centrifugal_load,
    set_material_properties,
    set_temperature_cycles,
)

fea_scripts = [
    os.path.join(os.path.dirname(__file__), "extract.py"),
]

static_base_input = os.path.join(os.path.dirname(__file__), "Disk-Universal-Static.inp")
base_input = os.path.join(os.path.dirname(__file__), "Disk-Universal.inp")

dim_mapping = {
    "U": ["disks", "op", "cycles", "fea_nodes", "coordinates"],
    "S": ["disks", "op", "cycles", "fea_cells", "stress_components"],
    "LE": ["disks", "op", "cycles", "fea_cells", "strain_components"],
    "MISES": ["disks", "op", "cycles", "fea_cells"],
    "NT": ["disks", "op", "cycles", "fea_nodes"],
    "PEEQ": ["disks", "op", "cycles", "fea_cells"],
    "time": ["disks", "op", "cycles"],
}

var_dims = {
    "normal": ["PEEQ", "time"],
    "refined": ["U", "S", "MISES", "NT", "LE", "PEEQ", "time"],
}


def dims_and_coords(disks, op_coords, variable, node_idx, cell_idx, cycles):
    """Create dimensions and coordinates for the output dataset."""
    m = dim_mapping[variable].copy()

    coords = {"disks": disks, "op": op_coords, "cycles": cycles}
    if "fea_nodes" in m:
        coords["fea_nodes"] = node_idx
    if "fea_cells" in m:
        coords["fea_cells"] = cell_idx
    if "stress_components" in m:
        coords["stress_components"] = ["S11", "S22", "S33", "S12"]
    if "strain_components" in m:
        coords["strain_components"] = ["LE11", "LE22", "LE33", "LE12"]
    if "coordinates" in m:
        coords["coordinates"] = ["x", "z"]

    return m, coords


def create_empty_results(
    dataset, op_coords, n_cycles=default_n_cycles, mode="plastic", refined=False
):
    """Create an empty dataset with the correct dimensions and coordinates."""
    node_idx, cell_idx = parse_mesh_from_input_file(base_input)
    if refined:
        output_division = 4
    else:
        output_division = 2

    nc = n_cycles if mode == "plastic" else 1
    cycles = (
        create_time_points(nc, np.array([1.0, 1.0, 1.0, 1.0]), output_division)[
            2
        ].astype(np.float32)
        / 4.0
    )

    disk_data = {}

    disk_data["errors"] = xr.DataArray(
        np.empty((len(dataset.disks), len(op_coords)), dtype=np.int8),
        coords={"disks": dataset.disks, "op": op_coords},
        dims=["disks", "op"],
    )

    vd = var_dims["refined" if refined else "normal"]

    for j in vd:
        if mode == "elastic" and j == "PEEQ":
            continue
        m, coords = dims_and_coords(
            dataset.disks,
            op_coords,
            j,
            node_idx,
            cell_idx,
            cycles,
        )

        empty_data_shape = tuple(len(coords[d]) for d in m)

        disk_data[j] = xr.DataArray(
            np.empty(empty_data_shape),
            coords=coords,
            dims=m,
        ).astype(np.float32)
    return xr.Dataset(disk_data)


def simulate(
    dataset,
    compute_path=None,
    op_coords=None,
    mode="plastic",
    refined=False,
    num_cpus=2,
    n_cycles=default_n_cycles,
):
    """Simulate the chunk of dataset of disk compositions using FEA at the
    specified operating conditions."""
    # Prepare helper variables
    node_idx, cell_idx = parse_mesh_from_input_file(base_input)

    # Process each disk individually
    fea_results = create_empty_results(dataset, op_coords, n_cycles, mode, refined)
    vd = var_dims["refined" if refined else "normal"]

    for i_d, d in enumerate(dataset.disks):
        disk = dataset.sel(disks=d)
        # Create experiment folder
        exp_name = f"disk_{d.disks.item()}"
        exp_path = os.path.join(compute_path, exp_name)
        os.makedirs(exp_path, exist_ok=True)
        os.chdir(exp_path)

        # Copy scripts
        for script in fea_scripts:
            os.system(f"cp {script} {exp_path}")

        dim_geom = dimensional_geometry(disk.geometry)
        r_o, r_i, L, thickness = [v.item() for v in dim_geom]

        heat_time = disk.operations.sel(parameters="heat_time")

        cooling_time = approximate_cooling_time(disk.properties, heat_time, L)
        cooling_time = cooling_time.to_numpy()

        mat_prop = disk.properties.to_numpy()

        dT = disk.operations.sel(parameters="delta_T").to_numpy()
        omega = disk.operations.sel(parameters="omega").to_numpy()
        heat_time = heat_time.to_numpy()

        # Prepare the input files for coupled conditions
        inp = inpRW.inpRW(
            base_input,
            preserveSpacing=True,
            useDecimal=True,
            organize=True,
        )
        inp.parse()
        inp.outputFolder = exp_path + os.path.sep

        # Adjust mesh and material properties one time
        scale_mesh(inp, r_o, r_i, thickness)
        set_material_properties(inp, mat_prop)

        # Prepare the input files for plastic conditions
        if mode == "elastic":
            # Prepare the input files for elastic conditions
            convert_to_elastic(inp)
            nc = 1
        else:
            nc = n_cycles

        output_division = 4 if refined else 2

        for i in range(disk.op.shape[0]):
            exp_op_name = f"{exp_name}-OP{i}-{mode.capitalize()}"

            odb_exists = os.path.exists(os.path.join(exp_path, f"{exp_op_name}.odb"))
            lck_exists = os.path.exists(os.path.join(exp_path, f"{exp_op_name}.lck"))
            sma_exists = os.path.exists(
                os.path.join(exp_path, f"{exp_op_name}.SMABulk")
            )
            if (
                odb_exists
                and not lck_exists
                and not sma_exists
                and (os.system(f"abaqus python extract.py is_empty {exp_op_name}") == 0)
            ):
                continue

            # Clean up all files for this OP
            os.system(f"rm -rf {exp_path}/{exp_op_name}*")

            l_cycle = default_cycle.copy()
            l_cycle[1] = heat_time[i]
            l_cycle[3] = cooling_time[i]

            set_temperature_cycles(inp, nc, l_cycle, dT[i], output_division)
            set_centrifugal_load(inp, omega[i])

            rename(inp, exp_op_name)
            inp.updateInp()
            inp.writeInp(output=inp.inpName)

            # Run the job
            os.system(
                f"abaqus job={exp_op_name} cpu={num_cpus} interactive ask_delete=OFF"
            )

        # Extract results from ODB
        os.system(
            "abaqus python extract.py all {exp_name} {n_op} {mode}".format(
                exp_name=exp_name, n_op=disk.op.shape[0], mode=mode.capitalize()
            )
        )

        try:
            # Collect results
            with open(os.path.join(exp_path, f"{exp_name}_results.pkl"), "rb") as f:
                results = pickle.load(f, encoding="latin1")
        except (pickle.UnpicklingError, FileNotFoundError) as e:
            print("Rescheduling")
            os.chdir(compute_path)
            os.system(f"rm -rf {exp_path}/{exp_name}*")
            raise Reschedule()

        try:
            results = results["results"]
            # Collect all the results
            fea_results["errors"][i_d] = np.array(
                [len(res["errors"]) for res in results], dtype=np.int8
            ).reshape(fea_results["errors"][i_d].shape)

            for j in vd:
                if mode == "elastic" and j == "PEEQ":
                    continue
                fea_results[j][i_d] = pad_arrays(
                    [res[j] for res in results],
                    fea_results[j][i_d].shape,
                    np.nan,
                )

        except Exception as e:
            error = f"Error in disk {d.disks.item()}: {e}"
            raise type(e)(error).with_traceback(e.__traceback__)

        # Clean up
        os.system(
            f"rm -f {exp_path}/{exp_name}*.com"
            + f" {exp_path}/{exp_name}*.dat"
            + f" {exp_path}/{exp_name}*.msg"
            + f" {exp_path}/{exp_name}*.prt"
            + f" {exp_path}/{exp_name}*.sta"
            + f" {exp_path}/{exp_name}*.env"
            + f" {exp_path}/*.py"
            + f" {exp_path}/{exp_name}_results.pkl"
        )
        os.chdir(compute_path)

    return fea_results


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
@click.option("--tmp", "-t", type=str, help="Temporary directory for processing")
@click.option("--mode", default="plastic")
@click.option("--refined", is_flag=True)
@click.option("--n-cycles", default=default_n_cycles, type=int)
@dask_cluster_options
def main(
    disks,
    cases,
    tmp,
    mode,
    refined,
    n_cycles,
    queue,
    n_cores,
    n_jobs,
    time_limit,
    network,
    network_sched,
):
    if queue == "localhost":
        client = Client()
    else:
        if network_sched is None:
            network_sched = network

        # Parse slurm time limit
        lifetime = f"{slurm_time_to_minutes_minus_five(time_limit)}m"
        cluster = SLURMCluster(
            cores=n_cores,
            job_cpu=n_cores * 2,
            processes=n_cores,
            memory=f"{6*n_cores*2}GB",
            job_mem=f"{3*n_cores*2}GB",
            interface=network,
            queue=queue,
            walltime=time_limit,
            worker_extra_args=["--lifetime", lifetime, "--lifetime-stagger", "2m"],
            scheduler_options={"interface": network_sched},
        )
        cluster.adapt(minimum_jobs=min(n_jobs // 2, 1), maximum_jobs=n_jobs)

        client = Client(cluster)
        print(client)
        print(client.dashboard_link)

    disks = xr.open_dataset(disks, engine="zarr")
    simulations = xr.open_dataset(cases, engine="zarr")

    combined = xr.merge([disks, simulations], join="inner")

    combined.chunk(
        {"disks": 10, "property": -1, "mat_cells": -1, "op": -1, "parameters": -1}
    ).map_blocks(
        simulate,
        kwargs=dict(
            compute_path=tmp,
            op_coords=combined.op,
            mode=mode,
            refined=refined,
            n_cycles=n_cycles,
        ),
    ).to_zarr(
        cases, mode="a"
    )


if __name__ == "__main__":
    main()
