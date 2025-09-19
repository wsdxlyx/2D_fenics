import os

import click
import numpy as np
import pandas as pd
import xarray as xr
from natsort import natsorted


def remove_units_from_columns(df):
    """
    Remove units from the columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns containing units.

    Returns
    -------
    pandas.DataFrame
        DataFrame with units removed from the column names.
    """
    df.columns = df.columns.str.replace(r"\s*\(.*\)\s*", "", regex=True)
    return df


def flatten_single_list_elements(df):
    """
    Flatten single list elements in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns containing lists.

    Returns
    -------
    pandas.DataFrame
        DataFrame with single list elements flattened.
    """
    for col in df.columns[df.dtypes == "object"]:
        if df[col].apply(lambda x: isinstance(x, str) and "[" in x).all():
            try:
                corrected = (
                    df[col]
                    .str.replace("[", "")
                    .str.replace("]", "")
                    .apply(lambda x: np.array(x, dtype=float))
                )
                df[col] = corrected
            except ValueError:
                continue
    return df


@click.command()
@click.argument("csv_folder", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="materials.nc")
def csv_to_xarray(csv_folder, output):
    columns_of_interest = [
        "Young's Modulus",
        "Poisson Ratio",
        "Yield Strength",
        "CTE",
        "Thermal Conductivity",
        "Density",
        "Specific Heat Capacity",
        "Enthalpy of Oxidation",
        "HCS",
        "CSC",
    ]
    correction = np.array([1e9, 1.0, 1e6, 1.0, 1.0, 1e3, 1.0, 1.0, 1.0, 1.0])

    columns_rename = {}
    columns_rename["Specific heat capacity"] = "Specific Heat Capacity"
    columns_rename["Heat Capacity"] = "Specific Heat Capacity"
    columns_rename["Enthalpy of Oxygen"] = "Enthalpy of Oxidation"
    columns_rename["Thermal Expansivity"] = "CTE"

    datasets = []

    for file in natsorted(os.listdir(csv_folder)):
        if not file.endswith(".csv"):
            continue
        trans_name = os.path.splitext(file)[0]
        prop = pd.read_csv(os.path.join(csv_folder, file))
        prop = remove_units_from_columns(prop)
        prop = flatten_single_list_elements(prop)
        prop = prop.rename(columns=columns_rename)
        prop["Poisson Ratio"] = 0.3
        prop.loc[:, columns_of_interest] *= correction

        conc_col = prop.columns[prop.columns.str.contains("Concentration")].values[0]

        if prop.columns.str.contains("wt%").any():
            mass_cols = prop.columns[prop.columns.str.contains("wt%")]
        else:
            mass_cols = prop.columns[prop.columns.str.endswith("mass_fraction")]

        composition = xr.DataArray(prop.loc[:, mass_cols], dims=["sample", "elements"])
        relative_concentration = xr.DataArray(
            prop[conc_col].values.flatten() / 100.0, dims=["sample"]
        )
        properties = xr.DataArray(
            prop.loc[:, columns_of_interest],
            dims=["sample", "property"],
            coords={"property": columns_of_interest, "transition": trans_name},
        )

        print(f"Loaded {file} with {len(prop)} samples and {len(mass_cols)} elements.")
        print(
            f"Composition: {composition.shape}, Relative Concentration: {relative_concentration.shape}, Properties: {properties.shape}"
        )
        x = xr.Dataset(
            {
                "composition": composition,
                "relative_concentration": relative_concentration,
                "properties": properties,
            }
        )

        datasets.append(x)

    ds = xr.concat(datasets, dim="sample")

    ds = ds.reset_index("sample", drop=True).assign_coords(
        sample=np.arange(len(ds.sample))
    )

    if output.endswith(".nc") or output.endswith(".h5"):
        ds.to_netcdf(output)
    else:
        ds.to_zarr(output)


if __name__ == "__main__":
    csv_to_xarray()
