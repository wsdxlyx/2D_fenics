"""
Adapted from Shen Wang's code
"""

import os
import pickle
import subprocess

import click
import inpRW
import numpy as np
import xarray as xr
from dask.distributed import Client, Reschedule
from dask_jobqueue import SLURMCluster

from mmdisk.materials.utils import approximate_cooling_time
from mmdisk.utils import dimensional_geometry

from ..common import cycle, n_cycles
from .prepare import (
    rename,
    scale_mesh,
    set_centrifugal_load,
    set_material_properties,
    set_temperature_cycles,
)

static_base_input = os.path.join(os.path.dirname(__file__), "Disk-Universal-Static.inp")
base_input = os.path.join(os.path.dirname(__file__), "Disk-Universal.inp")

fea_scripts = [
    os.path.join(os.path.dirname(__file__), "extract.py"),
]


def find_limit_for_disk(
    disk, compute_path=None, nn=3, num_cpus=2, T_tol=5.0, omega_tol=10.0
):

    if disk.sizes["disks"] == 0:
        return xr.Dataset(
            {
                "limits": xr.DataArray(
                    np.empty((1, 12, nn + 1)),
                    dims=["disks", "variables", "nn"],
                    coords={
                        "variables": [
                            "T_EL_low",
                            "T_EL_mean",
                            "T_EL_high",
                            "Omega_EL_low",
                            "Omega_EL",
                            "Omega_EL_high",
                            "T_SD_low",
                            "T_SD_mean",
                            "T_SD_high",
                            "Omega_SD_low",
                            "Omega_SD",
                            "Omega_SD_high",
                        ],
                        "nn": np.arange(nn + 1),
                    },
                )
            }
        )

    disk = disk.sel(disks=disk.disks[0].item())

    # Create experiment folder
    exp_name = f"disk_{disk.disks.item()}"
    exp_path = os.path.join(compute_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    os.chdir(exp_path)

    # Copy scripts
    for script in fea_scripts:
        os.system(f"cp {script} {exp_path}")

    dim_geom = dimensional_geometry(disk.geometry)
    r_o, r_i, L, thickness = [v.item() for v in dim_geom]

    if "heat_time" in disk:
        heat_time = disk.heat_time.item()
    else:
        heat_time = 20.0

    cooling_time = approximate_cooling_time(disk.properties, heat_time, L)
    cooling_time = cooling_time.to_numpy()

    mat_prop = disk.properties.to_numpy()
    # heat_time = heat_time.to_numpy()

    dT_0 = mat_prop[-1, 2] / (mat_prop[-1, 0] * mat_prop[-1, 3])  # Sy / (E * alpha)

    l_cycle = cycle.copy()
    l_cycle[1] = heat_time  # [0]
    l_cycle[3] = cooling_time  # [0]

    # Prepare the input files for cyclic conditions
    inp = inpRW.inpRW(
        base_input,
        preserveSpacing=True,
        useDecimal=True,
        organize=True,
    )
    inp.parse()
    inp.outputFolder = exp_path + os.path.sep
    scale_mesh(inp, r_o, r_i, thickness)
    set_material_properties(inp, mat_prop)

    # Search elastic limit
    omega = disk.max_speeds.sel(limit_kind="elastic").item()
    Omega_EL = [np.sqrt(1.0 * i / nn) * omega for i in range(nn + 1)]
    Omega_EL[0] = 0.00001

    Omega_EL_low = [i for i in Omega_EL]
    Omega_EL_high = [i for i in Omega_EL]

    Omega_EL_low[-1] = omega
    Omega_EL_high[-1] = omega

    T_start = dT_0 * 1.2
    T_EL_low = []
    T_EL_high = []
    T_EL_mean = []
    T = 0

    def run(T, name, nc):
        set_temperature_cycles(
            inp,
            nc,
            l_cycle,
            T,
        )
        rename(inp, name)
        inp.updateInp()
        inp.writeInp(output=inp.inpName)
        counter = 0
        while not os.path.exists(f"{name}.odb") and counter < 5:
            os.system(f"abaqus job={name} cpu={num_cpus} interactive ask_delete=OFF")
            counter += 1

        run_sp = subprocess.run(
            f"abaqus python extract.py single_peeq {name}",
            shell=True,
            check=True,
            capture_output=True,
        )
        time, peeq = pickle.loads(run_sp.stdout, encoding="latin-1")
        if len(time) != nc * l_cycle.shape[0] + 1:
            return -1
        elif (peeq <= 1e-8).all():
            return 0
        elif np.diff(peeq[(-2 * 8 - 1) :: 8], axis=0).max() <= 2e-5:
            return 1
        else:
            return 2

    for counter, omega in enumerate(Omega_EL[:-1]):
        if not T:
            T = T_start
        T_high = T_start
        T_low = 0
        flag = []

        set_centrifugal_load(inp, omega)

        while T_high - T_low >= T_tol or not (
            {0, 1}.issubset(flag) or {0, 2}.issubset(flag)
        ):
            T_old = T
            temp1 = run(T, f"{exp_name}-EL-{counter}-{len(flag) + 1}", 4)
            flag.append(temp1)
            if flag[-1] == 0:
                T_low = T
                if T_high == T_low:
                    T_high = max(T_high * 1.3, T + 100)
                if {1}.issubset(flag):
                    T = T + (T_high - T_low) / 2.0
                else:
                    T = T_high
            else:
                T_high = T
                T = T - (T_high - T_low) / 2.0

            if len(flag) > 15 and T_old < T_tol:
                break

        T_EL_low.append(T_low)
        T_EL_high.append(T_high)
        T_EL_mean.append(T)

    T_EL_low.append(0)
    T_EL_high.append(0)
    T_EL_mean.append(0)

    # Find shakedown limit
    # omega_high = max_speeds["max_plastic_speed"]
    # omega_low = max_speeds["max_elastic_speed"]
    # omega = (omega_high + omega_low) / 2.0

    # flag = []
    # # Time = []
    # # Critical3 = []

    # while omega_high - omega_low >= omega_tol or not (
    #     {1, 2}.issubset(flag) or {1, -1}.issubset(flag)
    # ):
    #     set_centrifugal_load(inp, omega)
    #     temp1 = run(0, f"{exp_name}-SD-w-{len(flag) + 1}", 2)
    #     flag.append(temp1)
    #     if flag[-1] == 0 or flag[-1] == 1:
    #         omega_low = omega
    #         if omega_high == omega_low:
    #             omega_high = 10000
    #         if {2}.issubset(flag) or {-1}.issubset(flag):
    #             omega = omega + (omega_high - omega_low) / 2.0
    #         else:
    #             omega = omega_high
    #     else:
    #         omega_high = omega
    #         omega = omega - (omega_high - omega_low) / 2.0

    #     if len(flag) > 1000:
    #         print("Couldn't find shakedown limit (DT=0)")
    #         print(flag)
    #         break
    omega = disk.max_speeds.sel(limit_kind="burst").item()

    Omega_SD = [pow(1.0 * i / nn, 1.0 / 2) * omega for i in range(nn + 1)]
    Omega_SD[0] = 0.00001

    Omega_SD_low = [i for i in Omega_SD]
    Omega_SD_high = [i for i in Omega_SD]

    Omega_SD_low[-1] = omega
    Omega_SD_high[-1] = omega

    T_start = dT_0 * 3.0
    T_SD_low = []
    T_SD_high = []
    T_SD_mean = []
    T = 0

    for counter, omega in enumerate(Omega_SD[:-1]):
        if not T:
            T = T_start
        else:
            T = 1.1 * T
        T_high = T_start
        if omega < Omega_EL[-1]:
            T_low = min(T_EL_low[:-1])
        else:
            T_low = 0
        flag = []

        set_centrifugal_load(inp, omega)

        while T_high - T_low >= T_tol or not (
            {1, 2}.issubset(flag) or {1, -1}.issubset(flag) or {0, 2}.issubset(flag)
        ):
            temp1 = run(T, f"{exp_name}-SD-{counter}-{len(flag) + 1}", n_cycles)
            flag.append(temp1)
            T_old = T

            if flag[-1] == 0 or flag[-1] == 1:
                T_low = T
                if T_high == T_low:
                    T_high = T_high + 3000
                if {2}.issubset(flag) or {-1}.issubset(flag):
                    T = T + (T_high - T_low) / 2.0
                else:
                    T = T_high
            else:
                T_high = T
                T = T - (T_high - T_low) / 2.0

            if len(flag) > 15 and T_old < T_tol:
                break

        T_SD_low.append(T_low)
        T_SD_high.append(T_high)
        T_SD_mean.append(T)

    T_SD_low.append(0)
    T_SD_high.append(0)
    T_SD_mean.append(0)

    x_sd = Omega_SD
    y_sd = T_SD_mean
    slope = [(y_sd[i + 1] - y_sd[i]) / (x_sd[i + 1] - x_sd[i]) for i in range(nn)]
    slopechange = [abs(slope[i + 1] - slope[i]) for i in range(len(slope) - 1)]
    ind_s = slopechange.index(max(slopechange)) + 1 + 1 - 1

    for ind in range(ind_s, nn):
        T_start = T_SD_low[ind]
        omega_start = Omega_SD[ind]

        omega = omega_start
        omega_high = omega_start
        omega_low = Omega_SD[ind - 1]
        flag = []

        while omega_high - omega_low >= omega_tol or not (
            {1, 2}.issubset(flag) or {1, -1}.issubset(flag) or {0, 2}.issubset(flag)
        ):
            set_centrifugal_load(inp, omega)
            temp1 = run(T_start, f"{exp_name}-SD-f{ind}-{len(flag) + 1}", n_cycles)
            flag.append(temp1)
            if flag[-1] == 0 or flag[-1] == 1:
                omega_low = omega
                if omega_high == omega_low:
                    omega_high = disk.max_speeds.sel(limit_kind="burst").item() * 1.2
                if {2}.issubset(flag) or {-1}.issubset(flag):
                    omega = omega + (omega_high - omega_low) / 2.0
                else:
                    omega = omega_high
            else:
                omega_high = omega
                omega = omega - (omega_high - omega_low) / 2.0

        Omega_SD[ind] = omega

        results = np.array(
            [
                T_EL_low,
                T_EL_mean,
                T_EL_high,
                Omega_EL_low,
                Omega_EL,
                Omega_EL_high,
                T_SD_low,
                T_SD_mean,
                T_SD_high,
                Omega_SD_low,
                Omega_SD,
                Omega_SD_high,
            ]
        )
        return xr.Dataset(
            {
                "limits": xr.DataArray(
                    results[None, ...],
                    dims=["disks", "variables", "nn"],
                    coords={
                        "variables": [
                            "T_EL_low",
                            "T_EL_mean",
                            "T_EL_high",
                            "Omega_EL_low",
                            "Omega_EL",
                            "Omega_EL_high",
                            "T_SD_low",
                            "T_SD_mean",
                            "T_SD_high",
                            "Omega_SD_low",
                            "Omega_SD",
                            "Omega_SD_high",
                        ],
                        "nn": np.arange(nn + 1),
                    },
                )
            }
        )


@click.command()
@click.option("--case", "-c", type=click.Path(), help="Input Zarr DB")
@click.option("--output", "-o", type=str, help="Output database")
@click.option("--tmp", "-t", type=str, help="Temporary directory for processing")
@click.option("--queue", default="newnodes")
@click.option("--n-cores", default=1)
@click.option("--n-jobs", default=30)
@click.option("--time-limit", default="12:00:00")
def find_limits(case, output, tmp, queue, n_cores, n_jobs, time_limit):
    if queue == "localhost":
        client = Client()
    else:
        cluster = SLURMCluster(
            cores=n_cores,
            job_cpu=n_cores * 2,
            processes=n_cores,
            memory=f"{3*n_cores*2}GB",
            interface="ib5",
            queue=queue,
            walltime=time_limit,
        )
        cluster.scale(jobs=n_jobs)

        client = Client(cluster)
        print(client)
        print(client.dashboard_link)

    ds = xr.open_zarr(case)

    if os.path.exists(output):
        print(f"Output file {output} already exists.")
        output_ds = xr.open_zarr(output)
        missing_disks = output_ds.disks[
            np.isnan(output_ds.limits).any(["nn", "variables"]).compute()
        ].load()
        zarr_args = {"mode": "r+", "region": "auto"}
    else:
        missing_disks = ds.disks.load()
        zarr_args = {"mode": "w-"}

    chunked = ds.sel(disks=missing_disks).chunk(
        {"disks": 1, "property": -1, "mat_cells": -1, "geom": -1}
    )
    chunked.map_blocks(
        find_limit_for_disk,
        kwargs=dict(compute_path=tmp, nn=4),
    ).to_zarr(output, **zarr_args)


if __name__ == "__main__":
    find_limits()
