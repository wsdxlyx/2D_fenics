#!/usr/bin/env python3
# simulate_FL.py – finite life (Nf) per (disk, op) using composition-dependent εf, c
# Driver mirrors find_limits_lb.py: batchwise region writes + "fully missing disk" rerun logic.

import gc, os
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

import click, numpy as np, xarray as xr
from mpi4py import MPI
from petsc4py import PETSc
from dask.graph_manipulation import bind
import dask  # keep import to match usage pattern

from mmdisk.cli_utils import (
    dask_cluster_options, get_continuous_chunks,
    start_dask_cluster_and_client, wait_for_futures, cleanup_limits_FL
)
from mmdisk.fea.common             import create_time_points, default_cycle, default_n_cycles
from mmdisk.fea.fenics.incremental  import ConvergenceError, incremental_cyclic_fea
from mmdisk.fea.fenics.mesh         import mesh_and_properties, mesh_max_size
from mmdisk.fea.fenics.utils        import get_loadfunc, get_time_list_alt
from mmdisk.fea.fenics.FL_utils     import fatigue_criter_range

# ──────────────────────────────── dim map ────────────────────────────────
_dim_mapping = {
    "u"        : ["disks","op","cycles","fea_nodes","coordinates"],
    "sig"      : ["disks","op","cycles","fea_cells","stress_components"],
    "T"        : ["disks","op","cycles","fea_nodes"],
    "PEEQ"     : ["disks","op","cycles","fea_cells"],
    "strain"   : ["disks","op","cycles","fea_cells","strain_components"],
    "Nf_maxP"  : ["disks","op"],
    "Nf_maxS"  : ["disks","op"],
    "Nf_BM"    : ["disks","op"],
    "Nf_VM"    : ["disks","op"],
    "eid_maxP" : ["disks","op"],
    "eid_maxS" : ["disks","op"],
    "eid_BM"   : ["disks","op"],
    "eid_VM"   : ["disks","op"],
}

def _clean(label) -> str:
    if isinstance(label, (bytes, bytearray)):
        label = label.decode("utf-8", errors="ignore")
    return label.rstrip("\x00").strip()

def _dims_and_coords(var, dataset, cycles, max_dim_cells, max_dim_nodes):
    dims   = _dim_mapping[var]
    coords = {"disks": dataset.disks, "op": dataset.op}
    if "cycles"            in dims: coords["cycles"] = cycles
    if "fea_nodes"         in dims: coords["fea_nodes"] = np.arange(max_dim_nodes)
    if "fea_cells"         in dims: coords["fea_cells"] = np.arange(max_dim_cells)
    if "stress_components" in dims: coords["stress_components"] = ["S11","S22","S33","S12"]
    if "coordinates"       in dims: coords["coordinates"] = ["x","z"]
    if "strain_components" in dims: coords["strain_components"] = ["eps_rr","eps_theta","eps_zz","eps_rz"]
    shape = tuple(len(coords[d]) for d in dims)
    return dims, coords, shape

# ───── composition-dependent εf, c (mapping recipe) ─────
BASE_PROPS = {"IN":dict(eps_f=0.22, c=-0.60),
              "INC":dict(eps_f=0.25, c=-0.60),
              "MK":dict(eps_f=0.33, c=-0.60)}
GROUP_TO_ALLOYS = {0:("IN","INC"), 1:("IN","MK"), 2:("MK","INC")}

def _map_eps_c_to_fe(ds_any, eps_map, c_map, Hc, Wc):
    spatial = [d for d in ds_any.dims if ds_any.sizes[d] > 3]
    for y in spatial:
        ry, remy = divmod(ds_any.sizes[y], Hc)
        if remy: continue
        for x in spatial:
            if x == y: continue
            rx, remx = divmod(ds_any.sizes[x], Wc)
            if not remx:
                e = np.repeat(np.repeat(eps_map, ry, 0), rx, 1).ravel()
                c = np.repeat(np.repeat(c_map,  ry, 0), rx, 1).ravel()
                return e, c
    base = Hc * Wc
    for d in spatial:
        k, rem = divmod(ds_any.sizes[d], base)
        if rem == 0 and k >= 1:
            return np.tile(eps_map.ravel(), k), np.tile(c_map.ravel(), k)
    raise ValueError("Cannot align εf/c maps", ds_any.sizes)

# ───── NPZ loader ─────
def _prepare_inc_model(inc_npz_path: str):
    if not os.path.exists(inc_npz_path):
        raise click.ClickException(f"INC model NPZ not found: {inc_npz_path}")
    dat = np.load(inc_npz_path, allow_pickle=True)
    if not all(k in dat.files for k in ("v0","v1")):
        raise click.ClickException(f"NPZ must contain 'v0' and 'v1'. Found: {list(dat.files)}")
    v0 = dat["v0"].astype("float32")
    v1 = dat["v1"].astype("float32")
    return {"v0": v0, "v1": v1}

def _eps_c_vectors_for_disk(sub_zarr_path, inc_model, disk_index, fea_cells_len):
    ds_sub = xr.open_zarr(sub_zarr_path)
    comp   = ds_sub["elemental_composition"].isel(disks=disk_index).load()
    Hc, Wc, _ = comp.shape

    v0, v1 = inc_model["v0"], inc_model["v1"]
    dV, dVn = v1 - v0, np.maximum((v1 - v0)**2, 1e-8).sum(1)

    flat = comp.values.reshape(-1, 10)
    proj = ((flat[:,None,:]-v0[None,:,:])*dV).sum(-1)/dVn
    proj = np.clip(proj, 0, 1)
    err  = ((v0[None,:,:]+proj[...,None]*dV-flat[:,None,:])**2).sum(-1)

    bestT = err.argmin(1); bestF = proj[np.arange(len(flat)), bestT]
    bad   = err.min(-1) > 2e-2; bestT[bad] = -1; bestF[bad] = np.nan

    group = np.full_like(bestT, -1)
    group[(0<=bestT)&(bestT<=9)]   = 0
    group[(10<=bestT)&(bestT<=19)] = 1
    group[(20<=bestT)&(bestT<=29)] = 2
    group = group.reshape(Hc, Wc); bestF = bestF.reshape(Hc, Wc)

    eps_f_map = np.full((Hc, Wc), np.nan, np.float32)
    c_map     = np.full((Hc, Wc), np.nan, np.float32)
    for g,(A,B) in GROUP_TO_ALLOYS.items():
        m = group == g
        if not m.any(): continue
        epsA,cA = BASE_PROPS[A]["eps_f"], BASE_PROPS[A]["c"]
        epsB,cB = BASE_PROPS[B]["eps_f"], BASE_PROPS[B]["c"]
        f = np.nan_to_num(np.clip(bestF[m], 0, 1))
        eps_f_map[m] = (1-f)*epsA + f*epsB
        c_map[m]     = (1-f)*cA  + f*cB

    dummy = xr.DataArray(np.empty((fea_cells_len,), np.float32), dims=["fea_cells"])
    eps_vec, c_vec = _map_eps_c_to_fe(dummy, eps_f_map, c_map, Hc, Wc)
    return eps_vec.astype(np.float32), c_vec.astype(np.float32)

# ───────────────────────────── FEA (single (disk, op)) ───────────────────
def simulate_single_case(
    disk_op, *, n_cycles, output_division, displacement_bc,
    strain_buf=None, other_outputs=None):

    mesh, props = mesh_and_properties(disk_op)

    ops1d   = disk_op.operations
    labels  = [_clean(p) for p in ops1d.coords["parameters"].values]
    values  = ops1d.values
    param   = dict(zip(labels, values))
    try:
        omega = float(param["omega"])
        dT    = float(param["delta_T"])
        t_hd  = float(param["heat_time"])
    except KeyError as e:
        raise ValueError(f"Missing parameter {e.args[0]!r} for "
                         f"disk={int(disk_op.coords['disks'].item())} "
                         f"op={int(disk_op.coords['op'].item())}")

    l_cycle = default_cycle.copy(); l_cycle[1], l_cycle[3] = t_hd, t_hd*20
    period,_,t_out = create_time_points(
        n_cycles, l_cycle, output_division=output_division, skip_cold=False)
    t_list = get_time_list_alt(
        l_cycle, l_cycle*np.array([0.25,0.1,0.25,0.01]),
        n_cycles, skip_cold=False)

    outputs = {} if other_outputs is None else other_outputs
    if strain_buf is not None:
        outputs["strain"] = strain_buf

    return incremental_cyclic_fea(
        mesh, props, get_loadfunc(dT,l_cycle), omega,
        t_list, t_out, period,
        Displacement_BC=displacement_bc,
        plastic_inverval=1,
        skip_cold=True,
        outputs=outputs,
    )

# ───────────────────────────── block kernel ──────────────────────────────
def simulate(
    dataset, *, n_cycles, displacement_bc, output_division,
    output_variables, max_dim_cells, max_dim_nodes, tmp_folder,
    sub_zarr_path, inc_model, amplitude_is_range):

    gc.collect(); PETSc.garbage_cleanup(MPI.COMM_SELF)

    pg,_,t_global = create_time_points(
        n_cycles, np.ones_like(default_cycle), output_division)
    t_global /= pg

    xr_ds = xr.Dataset({
        "errors": xr.DataArray(
            -np.ones((len(dataset.disks),len(dataset.op)), np.int8),
            dims=["disks","op"], coords={"disks":dataset.disks,"op":dataset.op})
    })
    xr_ds.errors.encoding = {"fill_value":-1}

    for v in output_variables:
        d,c,sh = _dims_and_coords(v, dataset, t_global, max_dim_cells, max_dim_nodes)
        xr_ds[v] = xr.DataArray(np.full(sh, np.nan, np.float32), dims=d, coords=c)
        xr_ds[v].encoding = {"fill_value": np.nan}

    need_life = any(v.startswith("Nf_") or v.startswith("eid_") for v in output_variables)
    half = 0.5 if amplitude_is_range else 1.0

    eps_c_cache = {}
    for i_d, d in enumerate(dataset.disks):
        # ── disk-level guard for eps/c mapping so a bad disk row doesn't kill the tile
        if need_life:
            di = int(d.item())
            try:
                if di not in eps_c_cache:
                    eps_vec, c_vec = _eps_c_vectors_for_disk(sub_zarr_path, inc_model, di, max_dim_cells)
                    eps_c_cache[di] = (eps_vec, c_vec)
                else:
                    eps_vec, c_vec = eps_c_cache[di]
            except Exception:
                # mark entire row as failed and continue with next disk
                xr_ds.errors.values[i_d, :] = -8  # custom negative code for eps/c mapping failure
                continue

        for i_op, op in enumerate(dataset.op):
            tmp_file = None
            if tmp_folder:
                tmp_file = os.path.join(tmp_folder, f"disk_{d.item()}_{op.item()}.npz")
                if os.path.exists(tmp_file):
                    dat = np.load(tmp_file, allow_pickle=True)
                    err = int(dat["errors"]) if "errors" in dat.files else -1
                    if err >= 0:
                        for var in output_variables:
                            if var in dat.files:
                                xr_ds[var].values[i_d, i_op] = dat[var]
                        xr_ds.errors[i_d, i_op] = err
                        continue
                    else:
                        try: os.remove(tmp_file)
                        except OSError: pass

            period,_,t_out = create_time_points(
                n_cycles, default_cycle, output_division, skip_cold=False)
            strain_buf = (np.empty((len(t_out), max_dim_cells, 4), np.float32)
                          if need_life else None)

            # ── broadened per-(disk,op) guard so any failure still returns a valid tile
            try:
                _,_,flag = simulate_single_case(
                    dataset.sel(disks=d, op=op),
                    n_cycles=n_cycles, output_division=output_division,
                    displacement_bc=displacement_bc,
                    strain_buf=strain_buf,
                    other_outputs={}
                )
                xr_ds.errors[i_d,i_op] = flag
            except (ConvergenceError, RuntimeError):
                xr_ds.errors[i_d,i_op] = -2
            except (KeyError, ValueError):
                xr_ds.errors[i_d,i_op] = -6
            except Exception:
                xr_ds.errors[i_d,i_op] = -9

            # ── life post-processing separated/guarded
            if need_life and xr_ds.errors[i_d, i_op] >= 0:
                try:
                    maxP, maxS, BM, VM = fatigue_criter_range(strain_buf, n_cycles)
                    A = {"maxP": maxP[-1]*half, "maxS": maxS[-1]*half, "BM": BM[-1]*half, "VM": VM[-1]*half}

                    def _nf_from_amp(amp):
                        N = np.full_like(amp, np.inf, np.float32)
                        ok = (amp > 0) & np.isfinite(eps_vec) & np.isfinite(c_vec)
                        N[ok] = 0.5 * (amp[ok] / eps_vec[ok]) ** (1.0 / c_vec[ok])
                        return N

                    N = {k: _nf_from_amp(v) for k, v in A.items()}
                    for key, Narr in N.items():
                        if np.isfinite(Narr).any():
                            j = int(np.nanargmin(Narr))
                            nf = float(Narr[j])
                            if f"Nf_{key}" in output_variables:
                                xr_ds[f"Nf_{key}"][i_d, i_op] = nf
                            if f"eid_{key}" in output_variables:
                                xr_ds[f"eid_{key}"][i_d, i_op] = j
                        else:
                            if f"Nf_{key}" in output_variables:
                                xr_ds[f"Nf_{key}"][i_d, i_op] = np.nan
                            if f"eid_{key}" in output_variables:
                                xr_ds[f"eid_{key}"][i_d, i_op] = -1
                except Exception:
                    # mark post-processing failure but keep the tile valid
                    for key in ("maxP","maxS","BM","VM"):
                        if f"Nf_{key}" in output_variables:
                            xr_ds[f"Nf_{key}"][i_d, i_op] = np.nan
                        if f"eid_{key}" in output_variables:
                            xr_ds[f"eid_{key}"][i_d, i_op] = -1
                    if xr_ds.errors[i_d, i_op] >= 0:
                        xr_ds.errors[i_d, i_op] = -4

            if tmp_file:
                np.savez_compressed(
                    tmp_file,
                    **{var: xr_ds[var].values[i_d, i_op] for var in output_variables},
                    errors=int(xr_ds.errors.values[i_d, i_op]),
                )
            del strain_buf; gc.collect()
    return xr_ds

# ─────────────── lightweight template block (for metadata creation/extension) ─────────
def _empty_results_block(block, *, output_variables):
    # Build a tiny ds with the right shape/dtypes, filled with NaNs / -1, no heavy work.
    D = int(block.sizes.get("disks", 0))
    O = int(block.sizes.get("op", 0))
    coords = {"disks": block.disks, "op": block.op}
    ds = xr.Dataset()
    ds["errors"] = xr.DataArray(np.full((D,O), -1, np.int8), dims=("disks","op"), coords=coords)
    for v in output_variables:
        ds[v] = xr.DataArray(np.full((D,O), np.nan, np.float32), dims=("disks","op"), coords=coords)
        ds[v].encoding = {"fill_value": np.nan}
    ds["errors"].encoding = {"fill_value": -1}
    return ds

# ═════════════════════════════ CLI / driver (LB-like) ════════════════════════════════
@click.command()
@click.option("-d","--disks",  type=click.Path(exists=True), required=True,
              help="SUB_ZARR (must contain elemental_composition)")
@click.option("-c","--cases",  type=click.Path(), required=True,
              help="Output Zarr path (must already contain 'operations').")
@click.option("-t","--tmp-folder", type=click.Path())
@click.option("--data-vars", multiple=True,
              default=["Nf_maxP","Nf_maxS","Nf_BM","Nf_VM"],
              help="Variables to store; add eid_* to also save worst element ids.")
@click.option("--inc-model", type=click.Path(exists=True), required=True,
              help="Path to NPZ with v0,v1.")
@click.option("--amplitude-is-range/--amplitude-is-amplitude", default=True)
@click.option("--output-division", default=2.0, type=float)
@click.option("--n-cycles", default=default_n_cycles, type=int)
@dask_cluster_options
def main(disks, cases, tmp_folder, data_vars, inc_model, amplitude_is_range,
         n_cycles, output_division, queue, n_cores, n_jobs, time_limit, network, network_sched):

    client = start_dask_cluster_and_client(
        queue, n_cores, n_jobs, time_limit,
        f"{n_cores*20}GB", f"{n_cores*10}GB",
        network, network_sched, ["export OMP_NUM_THREADS=1"])
    print(client); print(client.dashboard_link)

    if tmp_folder and not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # --- source disks & ops
    keep       = ["properties","heat_time","geometry","upper_density"]
    raw        = xr.open_zarr(disks)
    disks_ds   = raw[[v for v in keep if v in raw.data_vars]]

    # 'cases' must already contain operations coord
    sim_ds     = xr.open_zarr(cases)
    disks_ops  = xr.merge([disks_ds, sim_ds[["operations"]]], join="inner") \
                    .chunk({"disks":10,"op":-1})

    # For buffer alloc
    disk0      = disks_ds.isel(disks=0).load()
    _, max_cells = mesh_max_size(disk0)

    inc_model_dict = _prepare_inc_model(inc_model)

    sim_kw = dict(
        n_cycles=n_cycles,
        displacement_bc="fix-free",
        output_division=output_division,
        output_variables=list(data_vars),
        max_dim_cells=max_cells,
        max_dim_nodes=None,
        tmp_folder=tmp_folder,
        sub_zarr_path=disks,
        inc_model=inc_model_dict,
        amplitude_is_range=amplitude_is_range,
    )

    # Storage chunking (like LB driver)
    chunksize  = 10
    s_chunk_def = {"disks": chunksize, "op": -1}

    # ── Ensure result variables exist in 'cases' via a metadata-only write (LB style)
    existing = xr.open_zarr(cases)
    need_template = False
    for v in ["errors", *data_vars]:
        if v not in existing.data_vars:
            need_template = True
            break

    enc = {v: {"fill_value": np.nan} for v in data_vars}
    enc["errors"] = {"fill_value": -1}

    if need_template or existing.sizes.get("disks", 0) < disks_ops.sizes["disks"]:
        tmpl = disks_ops.map_blocks(_empty_results_block, kwargs=dict(output_variables=list(data_vars)))
        mode = "a"
        append_kwargs = {}
        if existing.sizes.get("disks", 0) < disks_ops.sizes["disks"]:
            append_kwargs = {"append_dim": "disks"}
        tmpl.chunk(s_chunk_def).to_zarr(cases, compute=False, mode=mode, encoding=enc, **append_kwargs)
        existing = xr.open_zarr(cases)

    # Check storage chunk size like LB (using a single representative var)
    check_var = "errors" if "errors" in existing.data_vars else (data_vars[0] if data_vars and data_vars[0] in existing.data_vars else None)
    if check_var is not None:
        dchunks = existing[check_var].chunksizes.get("disks", None)
        if dchunks and dchunks[0] != chunksize:
            raise ValueError(
                f"Output dataset has a different chunk size for disks ({dchunks[0]}) than desired ({chunksize})."
            )

    # ── Determine which disks are FULLY missing (LB style)
    missing_mask = None
    for v in data_vars:
        if v in existing:
            m = existing[v].isnull().all("op")
        else:
            m = xr.ones_like(existing["operations"].isel(op=0), dtype=bool)
        missing_mask = m if missing_mask is None else (missing_mask & m)
    if missing_mask is None:
        missing_mask = (existing["errors"] == -1).all("op")

    disks_ops = disks_ops  # (no-op; keep name)
    disks_missing = disks_ops.disks[ missing_mask.compute() ]

    # ── Batch, write with region='auto', cleanup bound (LB pattern)
    max_tasks  = n_cores * n_jobs
    zarr_args = {"mode": "r+", "region": "auto"}
    futures, prepared = [], 0
    batches = get_continuous_chunks(disks_missing.values, chunksize * (max_tasks // 2))
    print(f"There are {len(batches)} batches of disks.")

    for contiguous in batches:
        slc = disks_ops.sel(disks=contiguous)

        prior = slc.map_blocks(
            simulate, kwargs=sim_kw,
        ).chunk(s_chunk_def)

        store = prior.to_zarr(cases, compute=False, **zarr_args)

        fut = bind(
            cleanup_limits_FL(prior.disks, prior.op, tmp_folder=tmp_folder, file_format="npz"),
            store,
        ).compute(sync=False)

        fut.num_simulations = slc.sizes["disks"]
        futures.append(fut)

        prepared += slc.sizes["disks"]
        if prepared >= max_tasks * chunksize * 2:
            prepared -= wait_for_futures(futures)

    wait_for_futures(futures)
    client.close()

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
