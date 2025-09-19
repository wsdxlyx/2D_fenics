from typing import Dict, Optional, Union

import numpy as np
import xarray as xr


_molar_masses = {
    "Ni": 58.6934,
    "Cr": 51.9961,
    "Nb": 92.9064,
    "Mo": 95.95,
    "Ti": 47.867,
    "Al": 26.9815,
    "Co": 58.93,
    "Fe": 55.845,
    "Mn": 54.9380,
    "Si": 28.0855,
    "Cu": 63.546,
}


def molar_masses() -> xr.DataArray:
    mm_xr = xr.DataArray(
        list(_molar_masses.values()),
        dims=["elements"],
        coords={"elements": list(_molar_masses.keys())},
    )
    return mm_xr


def _strip_element_name(element: str) -> str:
    """Preprocess element names to remove any additional information."""
    for k in _molar_masses.keys():
        if k in element:
            return k

    return element


def thermal_effusivity(data):
    squared = (
        data["Thermal Conductivity"] * data["Specific Heat Capacity"] * data["Density"]
    )
    if hasattr(data, "pow"):
        return squared.pow(0.5)
    else:
        return np.sqrt(squared)


def approximate_cooling_time(
    properties: xr.DataArray,
    heat_time: Union[xr.DataArray | float],
    L: Union[xr.DataArray | float],
) -> xr.DataArray:
    therm_diff = (
        properties.sel(property="Thermal Conductivity")
        / properties.sel(property=["Specific Heat Capacity", "Density"]).prod(
            "property"
        )
    ).mean("mat_cells")

    depth_ratio = np.ceil(3.5 * np.sqrt(therm_diff * heat_time) / L * 10) / 10 * 2
    rL = depth_ratio.clip(max=1.0) * L

    cooling_time = np.ceil(np.log(160 / np.pi) * rL**2 / therm_diff / np.pi**2)
    return cooling_time


def convert_to_alloy_percents(ds: xr.DataArray, alloy_base: Optional[Dict] = None):
    if alloy_base is None:
        base_alloys = {}
        for tr in np.unique(ds.transitions):
            parts = tr.split("_")
            base_alloys[parts[0]] = None
            base_alloys[parts[1]] = None

        for i, alloy in enumerate(base_alloys.keys()):
            base_alloys[alloy] = np.zeros(len(base_alloys))
            base_alloys[alloy][i] = 1.0

    new_dims = ds.rel_concentration.dims + ("alloy",)

    ds["alloy_percents"] = xr.DataArray(
        ds["rel_concentration"].values[..., None].repeat(3, axis=-1),
        dims=new_dims,
        coords={
            **{dim: ds.coords[dim] for dim in ds.rel_concentration.dims},
            "alloy": list(base_alloys.keys()),
        },
    )

    def map_convert(group):
        parts = group.transitions.values[0].split("_")
        return (
            group.alloy_percents * base_alloys[parts[1]][:, None]
            + (1 - group.alloy_percents) * base_alloys[parts[0]][:, None]
        )

    ds["alloy_percents"] = (
        ds.groupby("transitions").apply(map_convert).transpose(..., "alloy")
    )
    return ds


def alloy_molar_mass(mass_fractions: xr.DataArray):
    """
    Convert mass fractions to mole fractions."
    """

    mm_xr = molar_masses().reindex(elements=mass_fractions.elements)
    mm_sum = (mass_fractions / mm_xr).sum("elements")

    if "Ni" not in mass_fractions.elements:
        mm_sum = mm_sum + (1 - mass_fractions.sum("elements")) / _molar_masses["Ni"]

    return 1 / mm_sum


def mass_to_mole_fractions(mass_fractions: xr.DataArray):
    """
    Convert mass fractions to mole fractions."
    """
    mm_xr = molar_masses()
    alloy_M = alloy_molar_mass(mass_fractions)
    return mass_fractions / mm_xr * alloy_M
