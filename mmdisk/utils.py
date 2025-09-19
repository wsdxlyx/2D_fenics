import numpy as np


def dimensional_geometry(geom):
    r_o = geom.sel(geom="ro")
    r_i = r_o * geom.sel(geom="ri_ro")
    L = r_o - r_i
    thickness = geom.sel(geom="thickness")
    return r_o, r_i, L, thickness


def pad_arrays(arrays, coord_shapes, pad_value=0):
    # Create a list to store the padded arrays
    padded_arrays = []

    for arr in arrays:
        if len(arr.shape) != len(coord_shapes[1:]):
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
        # Pad the current array with the specified padding value
        pad_width = [(0, c - a) for c, a in zip(coord_shapes[1:], arr.shape)]
        padded_array = np.pad(
            arr, pad_width, mode="constant", constant_values=pad_value
        )
        padded_arrays.append(padded_array)

    # Stack the padded arrays to create a unified numpy array
    unified_array = np.stack(padded_arrays)

    return unified_array
