import functools
import os
from datetime import timedelta
from typing import List, Optional

import click
import dask
from dask.distributed import Client, as_completed


def dask_cluster_options(func):
    @click.option("--queue", default="localhost")
    @click.option("--n-cores", default=1)
    @click.option("--n-jobs", default=5)
    @click.option("--time-limit", default="1-12:00:00")
    @click.option("--network", default=None)
    @click.option("--network-sched", default=None)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def slurm_time_to_minutes_minus_five(time_str: str) -> int:
    # Initialize time components
    days = 0
    hours = 0
    minutes = 0
    seconds = 0

    # Check if there's a day component
    if "-" in time_str:
        day_part, time_part = time_str.split("-", 1)
        days = int(day_part)
    else:
        time_part = time_str

    # Split the remaining part by ":"
    parts = time_part.split(":")

    # Determine which parts we got
    if len(parts) == 3:
        # Format: HH:MM:SS
        hours, minutes, seconds = map(int, parts)
    elif len(parts) == 2:
        # Format: MM:SS
        minutes, seconds = map(int, parts)
    elif len(parts) == 1:
        # Format: SS
        seconds = int(parts[0])
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    # Create a timedelta from the parsed components
    td = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    # Convert total duration to minutes
    total_minutes = int(td.total_seconds() // 60)

    # Subtract 5 minutes
    return total_minutes - 5


def start_dask_cluster_and_client(
    queue: str,
    n_cores: int,
    n_jobs: int,
    time_limit: str,
    memory: str,
    job_memory: str,
    network: str,
    network_sched: Optional[str] = None,
    prologue: List[str] = [],
) -> Client:
    if queue == "localhost":
        from dask.distributed import LocalCluster

        cluster = LocalCluster(
            n_workers=n_cores * n_jobs, threads_per_worker=1, processes=True
        )
    else:
        from dask_jobqueue import SLURMCluster

        from mmdisk.cli_utils import slurm_time_to_minutes_minus_five

        if network_sched is None:
            network_sched = network

        # Parse slurm time limit
        lifetime = f"{slurm_time_to_minutes_minus_five(time_limit)}m"
        job_extra_directives = []
        if "preempt" in queue:
            job_extra_directives.append("--requeue")
        cluster = SLURMCluster(
            cores=n_cores,
            job_cpu=n_cores,
            processes=n_cores,
            memory=memory,
            job_mem=job_memory,
            interface=network,
            queue=queue,
            walltime=time_limit,
            worker_extra_args=["--lifetime", lifetime, "--lifetime-stagger", "5m"],
            scheduler_options={"interface": network_sched},
            job_script_prologue=prologue,
            job_extra_directives=job_extra_directives,
        )
        cluster.scale(jobs=n_jobs)

    return Client(cluster)


def get_continuous_chunks(indices: list, max_size: int) -> list:
    """Get continuous chunks from a list of indices.

    Parameters
    ----------
    indices : list
        List of indices.

    Returns
    -------
    list: List of slices representing continuous chunks.
    """
    if len(indices) == 0:
        return []

    # Optionally, sort the list if not sorted:
    indices = sorted(indices)

    chunks = []
    start = indices[0]
    end = indices[0]

    for index in indices[1:]:
        if index == end + 1 and index - start < max_size:
            # Extend the current chunk
            end = index
        else:
            # Save the current chunk and start a new one
            chunks.append(slice(start, end))
            start = index
            end = index
    # Add the last chunk
    chunks.append(slice(start, end))

    return chunks


@dask.delayed
def cleanup_limits(*dims, tmp_folder=None, file_format="npy"):
    """Remove temporary files used for storing limits."""
    if tmp_folder is None:
        return

    for n in zip(*dims):
        identifier = "_".join(str(i) for i in n)
        tmp_file = os.path.join(tmp_folder, f"disk_{identifier}.{file_format}")
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


def wait_for_futures(futures):
    """Wait for futures to complete and release resources."""
    received = 0
    if len(futures) > 0:
        for future in as_completed(futures):
            future.result()
            future.release()
            received += future.num_simulations
        futures.clear()
    return received

@dask.delayed
def cleanup_limits_FL(*dims, tmp_folder=None, file_format="npz", prefix="disk"):
    """
    Remove temporary files named like:
        {prefix}_{d0}_{d1}_..._{dk}.{file_format}
    for every combination in the cartesian product of dims.

    Examples:
      cleanup_limits([0,1,2], tmp_folder=..., file_format="npy")
        → disk_0.npy, disk_1.npy, disk_2.npy

      cleanup_limits([10,11], range(30), tmp_folder=..., file_format="npz")
        → disk_10_0.npz, disk_10_1.npz, ..., disk_11_29.npz
    """
    if tmp_folder is None:
        return

    # Normalize inputs to 1-D Python lists
    norm = []
    for d in dims:
        # accept numpy/xarray/dask arrays
        try:
            d = np.asarray(d).ravel().tolist()
        except Exception:
            d = list(d)
        norm.append(d)

    # If no dims given, nothing to do
    if not norm:
        return

    for combo in product(*norm):
        identifier = "_".join(str(int(x)) for x in combo)  # int() guards 0-d arrays
        path = os.path.join(tmp_folder, f"{prefix}_{identifier}.{file_format}")
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            # best-effort; ignore races/permissions on cleanup
            pass
