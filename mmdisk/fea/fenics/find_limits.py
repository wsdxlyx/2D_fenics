"""
Adapted from Shen Wang's code for FEniCS
"""

import os

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "2"

import click
import dask.array as da
import numpy as np
import xarray as xr
from dask.distributed import Client
from mpi4py import MPI
from petsc4py import PETSc

from mmdisk.cli_utils import dask_cluster_options
from mmdisk.fea.common import create_time_points, default_cycle, default_n_cycles
from mmdisk.fea.fenics.incremental import ConvergenceError, incremental_cyclic_fea
from mmdisk.fea.fenics.mesh import get_properties, load_universal_mesh, scale_mesh
from mmdisk.fea.fenics.utils import (
    get_loadfunc,
    get_time_list_alt,
)
from mmdisk.materials.utils import approximate_cooling_time
from mmdisk.utils import dimensional_geometry


def find_limit_for_disk(
    disks, nn=3, T_tol=5.0, omega_tol=5.0, n_cycles=default_n_cycles
):
    xr_ds = xr.Dataset(
        {
            "limits": xr.DataArray(
                np.full((len(disks.disks), 12, nn + 1), np.nan),
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
                    "disks": disks.disks,
                },
            )
        }
    )

    if disks.sizes["disks"] == 0:
        return xr_ds

    disk = disks.sel(disks=disks.disks[0].item())

    dim_geom = dimensional_geometry(disk.geometry)
    r_o, r_i, L, thickness = [v.item() for v in dim_geom]

    if "heat_time" in disk:
        heat_time = disk.heat_time.item()
    else:
        heat_time = 20.0

    t_cooldwell = approximate_cooling_time(disk.properties, heat_time, L)
    t_cooldwell = t_cooldwell.item()
    t_cooldwell = np.ceil(t_cooldwell / 4) * 4

    mat_prop = disk.properties.to_numpy()
    dT_0 = mat_prop[-1, 2] / (mat_prop[-1, 0] * mat_prop[-1, 3])

    l_cycle = default_cycle.copy()
    l_cycle[1] = heat_time  # [0]
    l_cycle[3] = t_cooldwell  # [0]

    time_steps = l_cycle.copy() * [0.5, 0.1, 0.5, 0.25]
    period, _, t_output = create_time_points(n_cycles, l_cycle, output_division=0.25)
    t_list = get_time_list_alt(l_cycle, time_steps, n_cycles)

    mesh = load_universal_mesh()
    mesh = scale_mesh(mesh, r_o, r_i, thickness)
    props = get_properties(mesh, mat_prop)

    max_speeds = disk.max_speeds

    # Search elastic limit
    omega = max_speeds.sel(limit_kind="elastic").item()
    Omega_EL = [np.sqrt(1.0 * i / nn) * omega for i in range(nn + 1)]
    Omega_EL[0] = 0.00001

    Omega_EL_low = [i for i in Omega_EL]
    Omega_EL_high = [i for i in Omega_EL]
    Omega_EL_low[-1] = omega
    Omega_EL_high[-1] = omega

    T_start = 500
    T_EL_low = []
    T_EL_high = []
    T_EL_mean = []
    T = 0

    def run(T: float, omega: float) -> int:
        load = get_loadfunc(T, l_cycle)

        try:
            (_, peeq, flag) = incremental_cyclic_fea(
                mesh,
                props,
                load,
                omega,
                t_list,
                t_output,
                period,
                Displacement_BC="fix-free",
            )
        except ConvergenceError as e:
            flag = -1
            peeq = e.peeq

        if flag == 1:
            # Elastic
            return 0
        elif flag == 2:
            # Shakedown
            return 1
        elif flag == 0:
            # Full simulations completed
            delta_peeq = np.diff(peeq, axis=0)
            if (delta_peeq[-2:] < 2e-5).all():
                return 1
            else:
                return 2
        else:
            # Manage errors
            rpeeq = peeq[~np.isnan(peeq).any(axis=1)]
            if len(rpeeq) <= 2:
                return 2

            delta_peeq = np.diff(rpeeq, axis=0)
            if (delta_peeq[-2:] < 2e-5).all():
                return 1
            elif len(rpeeq) > 20:
                return 2

            return -1

    for omega in Omega_EL[:-1]:
        T_high = dT_0 * 1.5
        T_low = 0
        flag = []

        while T_high - T_low >= T_tol:
            T_load = (T_high + T_low) / 2.0
            temp1 = run(T_load, omega)
            flag.append(temp1)
            if flag[-1] == 0:
                T_low = T_load
            else:
                T_high = T_load

            if len(flag) > 20:
                break

        T_EL_low.append(T_low)
        T_EL_high.append(T_high)
        T_EL_mean.append(T_load)

    T_EL_low.append(0)
    T_EL_high.append(0)
    T_EL_mean.append(0)

    # Find shakedown limit
    omega = max_speeds.sel(limit_kind="burst").item()
    Omega_SD = [pow(1.0 * i / nn, 1.0 / 2) * omega for i in range(nn + 1)]
    Omega_SD[0] = 0.00001

    Omega_SD_low = [i for i in Omega_SD]
    Omega_SD_high = [i for i in Omega_SD]

    Omega_SD_low[-1] = omega
    Omega_SD_high[-1] = omega

    T_SD_low = []
    T_SD_high = []
    T_SD_mean = []

    for omega in Omega_SD[:-1]:
        if omega < Omega_EL[-1]:
            T_low = T_EL_low[0]
        else:
            T_low = 0
        T_high = dT_0 * 3
        flag = []

        while T_high - T_low >= T_tol:
            T_load = (T_high + T_low) / 2.0
            temp1 = run(T_load, omega)
            flag.append(temp1)

            if flag[-1] == 0 or flag[-1] == 1:
                T_low = T_load
            elif flag[-1] == 2:
                T_high = T_load
            else:
                T_high = T_high * 0.99

        T_SD_low.append(T_low)
        T_SD_high.append(T_high)
        T_SD_mean.append(T_load)

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

        while omega_high - omega_low >= omega_tol:
            temp1 = run(T_start, omega)
            flag.append(temp1)
            if flag[-1] == 0 or flag[-1] == 1:
                omega_low = omega
                if omega_high == omega_low:
                    omega_high = max_speeds.sel(limit_kind="burst").item()
                omega = omega + (omega_high - omega_low) / 2.0
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
                    "disks": disks.disks,
                },
            )
        }
    )


@click.command()
@click.option("--case", "-c", type=click.Path(), help="Input Zarr DB")
@click.option("--output", "-o", type=str, help="Output database")
@click.option("--queue", default="newnodes")
@click.option("--n-cores", default=1)
@click.option("--n-jobs", default=30)
@click.option("--time-limit", default="12:00:00")
@click.option("--nn", default=3)
def find_limits(case, output, nn, queue, n_cores, n_jobs, time_limit):
    if queue == "localhost":
        from dask.distributed import LocalCluster

        cluster = LocalCluster(
            n_workers=n_cores * n_jobs, threads_per_worker=1, processes=True
        )
    else:
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            cores=n_cores,
            job_cpu=n_cores * 2,
            processes=n_cores,
            memory=f"{3 * n_cores * 2}GB",
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
        kwargs=dict(nn=nn),
    ).to_zarr(output, **zarr_args)


if __name__ == "__main__":
    find_limits()
