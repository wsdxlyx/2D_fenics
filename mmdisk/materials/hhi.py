import numpy as np

from mmdisk.materials.utils import _strip_element_name

_elemental_HHI = {
    "Ni": 0.2295,
    "Cr": 0.1361,
    "Fe": 0.0916,
    "Nb": 0.0532,
    "Al": 0.28,
    "Ti": 0.7501,
    "Co": 0.1163,
    "Mo": 0.235,
    "Mn": 0.123,
    "Si": 0.1343,
    "Cu": 0.459,
}


def material_criticality(compositions, element_order=None):
    if element_order is None:
        element_order = ["Ti", "Cr", "Fe", "Mn", "Si", "Mo", "Al", "Nb", "Cu", "Co"]
    else:
        element_order = [_strip_element_name(el) for el in element_order]

    last_axis = len(compositions.shape) - 1

    weights = np.array([_elemental_HHI[el] for el in element_order])

    Ni_hhi = _elemental_HHI["Ni"] * (1.0 - np.sum(compositions, axis=last_axis))

    return np.sum(compositions * weights, axis=last_axis) + Ni_hhi
