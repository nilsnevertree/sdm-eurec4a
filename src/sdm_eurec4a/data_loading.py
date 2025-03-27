"""
This module contains functions to load data from the CLEO output files.
"""

from pathlib import Path
from typing import Literal, Tuple, Union
import matplotlib.pyplot as plt

import numpy as np
import scipy.interpolate
import xarray as xr
from sdm_eurec4a.constants import WaterConstants
from sdm_eurec4a.reductions import interpolate_dataset, mean_and_stderror_of_mean_dataset
from sdm_eurec4a.visulization import adjust_lightness_array, adjust_lightness
from sdm_eurec4a.constants import TimeSlices

from itertools import combinations


# eulerian_data_path = lambda data_dir, microphysics: data_dir / Path(
#     f"{microphysics}/combined/eulerian_dataset_combined.nc"
# )
# conservation_data_path = lambda data_dir, microphysics: data_dir / Path(
#     f"{microphysics}/combined/conservation_dataset_combined.nc"
# )
def __eulerian_data_path__(data_dir: Path, microphysic: str) -> Path:
    return data_dir / Path(f"{microphysic}/combined/eulerian_dataset_combined.nc")


def __conservation_data_path__(data_dir: Path, microphysic: str) -> Path:
    return data_dir / Path(f"{microphysic}/combined/conservation_dataset_combined.nc")


def __post_process_conservation_dataset__(
    ds: xr.Dataset, da_surface_area: xr.DataArray, timestep: float
) -> xr.Dataset:
    """
    Post-process the conservation dataset to have the correct units and add additional variables.
    """

    ds["source"].attrs["long_name"] = "Column Integrated Evaporation"
    ds["inflow"].attrs["long_name"] = "Cloud Base Precipitation Flux"
    ds["outflow"].attrs["long_name"] = "Surface Precipitation Flux"

    for var, new_var in zip(
        ["source", "inflow", "outflow", "reservoir_change"],
        [
            "source_precipitation",
            "inflow_precipitation",
            "outflow_precipitation",
            "reservoir_change_precipitation",
        ],
    ):
        attrs = ds[var].attrs.copy()
        # from  kg per dT per domain area
        # dT = 2s

        # # to    g per h per m^2
        # ds[var] = ds[var] / 2 * 3600 / ds['surface_area'] * 1e6
        # ds[var].attrs.update(attrs)
        # ds[var].attrs['units'] = 'mg/m^2/h'

        ds["surface_area"] = da_surface_area

        # to    mm / h
        ds[new_var] = ds[var] / 2 * 3600 / ds["surface_area"]  # kg / m^2 / h
        ds[new_var] = 1e3 * ds[new_var] / WaterConstants.density  # mm / h
        ds[new_var].attrs.update(attrs)
        ds[new_var].attrs["units"] = r"mm \, h^{-1}"

    for var, new_var in zip(
        ["source", "inflow", "outflow", "reservoir_change"],
        ["source_energy", "inflow_energy", "outflow_energy", "reservoir_change_energy"],
    ):
        attrs = ds[var].attrs.copy()
        # from  kg per dT per domain area
        # dT = 2s

        # # to    g per h per m^2
        # ds[var] = ds[var] / 2 * 3600 / ds['surface_area'] * 1e6
        # ds[var].attrs.update(attrs)
        # ds[var].attrs['units'] = 'mg/m^2/h'

        # to    mm / h
        ds[new_var] = ds[var] / timestep / ds["surface_area"]  # kg / m^2 / s
        ds[new_var] = ds[new_var] * WaterConstants.vapourization_heat  # J/s = W / m^2
        ds[new_var].attrs.update(attrs)
        ds[new_var].attrs["units"] = r"W \, m^{-2}"

    return ds


def __post_process_eulerian_dataset__(ds: xr.Dataset) -> xr.Dataset:

    ds["max_gridbox"] = ds["max_gridbox"].fillna(ds["gridbox"].max())

    # convert the liquid water content to g/m^3
    ds["liquid_water_content"] = 1e3 * ds["liquid_water_content"]
    ds["liquid_water_content"].attrs["units"] = "g/m^3"
    ds["liquid_water_content"].attrs["long_name"] = "Rain Water Content"

    # select the cloud liquid water content
    ds["cloud_liquid_water_content"] = ds["liquid_water_content"].sel(gridbox=ds["max_gridbox"])
    ds["cloud_liquid_water_content"].attrs["long_name"] = "Cloud Rain Water Content"

    # create variables for the mean radius of the distributions

    # ds["cloud_xi_radius_mean"] = ds["xi_radius_mean"].sel(gridbox=ds["max_gridbox"])
    # ds["cloud_xi_radius_mean"].attrs["long_name"] = "Cloud mean radius"
    # ds["cloud_xi_radius_mean"].attrs["units"] = "µm"

    # ds["small_cloud_xi_radius_mean"] = ds["small_xi_radius_mean"].sel(gridbox=ds["max_gridbox"])
    # ds["small_cloud_xi_radius_mean"].attrs["long_name"] = "Cloud mean radius"
    # ds["small_cloud_xi_radius_mean"].attrs["units"] = "µm"

    ds["cloud_mass_radius_mean"] = ds["mass_radius_mean"].sel(gridbox=ds["max_gridbox"])
    ds["cloud_mass_radius_mean"].attrs["long_name"] = "Cloud Mean Mass Radius"
    ds["cloud_mass_radius_mean"].attrs["units"] = "µm"

    # ds["small_cloud_mass_radius_mean"] = ds["small_mass_radius_mean"].sel(gridbox=ds["max_gridbox"])
    # ds["small_cloud_mass_radius_mean"].attrs["long_name"] = "Cloud mean mass radius"
    # ds["small_cloud_mass_radius_mean"].attrs["units"] = "µm"
    # from kg/m^3/s
    # # to mg/m^3/h
    # ds['evaporation_rate']  = - ds['massdelta_condensation'] * 1e6 * 3600
    # ds['evaporation_rate'].attrs['units'] = 'mg/m^3/h'
    # ds['evaporation_rate'].attrs['long_name'] = 'Evaporation rate'

    # from kg/m^3/s
    # to mm/m/h
    ds["evaporation_rate"] = -1e3 / WaterConstants.density * ds["massdelta_condensation"] * 3600
    ds["evaporation_rate"].attrs["units"] = r"mm \, h^{-1} \, m^{-1}"
    ds["evaporation_rate"].attrs["long_name"] = "Evaporation Rate"

    # from kg / m^3 / s
    # to W / m^3
    ds["evaporation_rate_energy"] = (
        ds["massdelta_condensation"] * WaterConstants.vapourization_heat
    )  # J/s / m^3 = W / m^3
    ds["evaporation_rate_energy"] = 1e3 * ds["evaporation_rate_energy"]  # mW / m^3
    ds["evaporation_rate_energy"].attrs["units"] = r"mW \, m^{-3}"
    ds["evaporation_rate_energy"].attrs["long_name"] = "Evaporation Rate"

    return ds


class MicrophysicsStyles:

    available_setups = (
        # "null_microphysics",
        "condensation",
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    )
    all_setups = (
        "null_microphysics",
        "condensation",
        "collision_condensation",
        "coalbure_condensation_small",
        "coalbure_condensation_large",
    )

    def __init__(self):

        self.available_setups = MicrophysicsStyles.available_setups

        markers = dict(
            null_microphysics="$*$",
            condensation="1",
            collision_condensation="x",
            coalbure_condensation_small="+",
            coalbure_condensation_large="2",
        )
        descriptions = dict(
            null_microphysics="No microphysics",
            condensation="Evaporation only",
            collision_condensation="Evaporation and Coalescence",
            coalbure_condensation_small="Evaporation and Coalescence Breakup and Rebound with small number of fragments",
            coalbure_condensation_large="Evaporation and Coalescence Breakup and Rebound with small number of fragments",
        )
        names = dict(
            null_microphysics="NullMicro",
            condensation="EvapOnly",
            collision_condensation="EvapCoal",
            coalbure_condensation_small="EvapCoalBuRe-few",
            coalbure_condensation_large="EVapCoalBuRe-many",
        )

        colors = dict(
            null_microphysics="grey",
            condensation="purple",
            collision_condensation="blue",
            coalbure_condensation_small="red",
            coalbure_condensation_large="orange",
        )

        dark_colors = {k: adjust_lightness(v, 0.75) for k, v in colors.items()}
        light_colors = {k: adjust_lightness(v, 1.25) for k, v in colors.items()}

        self.microphysics_styles = dict()

        for key in MicrophysicsStyles.available_setups:
            self.microphysics_styles[key] = dict(
                name=names[key],
                description=descriptions[key],
                linestyle="None",
                marker=markers[key],
                color=colors[key],
                dark_color=dark_colors[key],
                light_color=light_colors[key],
            )

    def __validate_key__(self, key: str):
        if key not in self.microphysics_styles.keys():
            raise ValueError(f"Unknown microphysics style {key}")

    def __getitem__(self, key: str):
        self.__validate_key__(key)
        return self.microphysics_styles[key]

    def __iter__(self):
        for key in self.microphysics_styles.keys():
            yield key

    def get_setup(
        self,
        key: Literal[
            "null_microphysics",
            "condensation",
            "collision_condensation",
            "coalbure_condensation_small",
            "coalbure_condensation_large",
        ],
    ):
        return self[key]

    def get_style(
        self,
        key: Literal[
            "null_microphysics",
            "condensation",
            "collision_condensation",
            "coalbure_condensation_small",
            "coalbure_condensation_large",
        ],
        colortype: Literal["normal", "dark", "light"] = "normal",
    ) -> dict:

        self.__validate_key__(key)

        style = dict(
            label=self[key]["name"],
            linestyle=self[key]["linestyle"],
            marker=self[key]["marker"],
        )

        if colortype not in ["normal", "dark", "light"]:
            raise ValueError(f"Unknown color type {colortype}")
        elif colortype == "normal":
            style["color"] = self[key]["color"]
        elif colortype == "dark":
            style["color"] = self[key]["dark_color"]
        elif colortype == "light":
            style["color"] = self[key]["light_color"]

        return style


__microphysics__ = tuple(MicrophysicsStyles())


class CleoDataset:

    def __init__(
        self,
        data_dir: Union[str, Path],
        microphysics: Tuple[str, ...] = __microphysics__,
        time_slice: slice = TimeSlices.quasi_stationary_state,
    ):
        self.variables_to_load = [
            "air_temperature",
            "gridbox",
            "gridbox_bottom",
            "gridbox_coord3",
            "gridbox_top",
            "gridbox_volume",
            "gridbox_coord3_norm",
            "liquid_water_content",
            "mass_radius_mean",
            "mass_radius_std",
            "mass_represented_temporal_mean",
            "massdelta_condensation",
            "max_gridbox",
            "precipitation",
            "pressure",
            "radius_bins",
            "relative_humidity",
            "specific_mass_vapour",
            "surface_area",
            "xi_radius_mean",
            "xi_radius_std",
            "xi_temporal_mean",
        ]

        self.data_dir = Path(data_dir)
        self.microphysics = microphysics
        self.time_slice = time_slice

        # load the data as the temporal mean and the standard error of the mean over the
        # provided time slice
        ds, ds_sem = self.__load_data__()
        # post process each dataset to have the correct units, and add additional variables
        # like the cloud liquid water content
        ds = self.__post_process_dataset__(ds)
        ds_sem = self.__post_process_dataset__(ds_sem)

        self.ds = ds
        self.ds_sem = ds_sem

        # add the evaporation fraction to both datasets.
        # The function also calculates the error propagation for the evaporation fraction for the
        # standard error of the mean dataset
        self.calculate_evaporation_fraction()

    def __eulerian_data_path__(self, microphysic: str) -> Path:
        return self.data_dir / Path(f"{microphysic}/combined/eulerian_dataset_combined.nc")

    def __conservation_data_path__(self, microphysic: str) -> Path:
        return self.data_dir / Path(f"{microphysic}/combined/conservation_dataset_combined.nc")

    def __load_data__(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Method to load the CLEO data from the output files.

        Returns
        -------
        Tuple[xr.Dataset, xr.Dataset]
            Tuple containing the dataset and the standard error of the mean dataset.
        """

        # Validate that all datasets are available
        ## Data loading and timestepping analysis
        for mp in self.microphysics:
            try:
                ds = xr.open_dataset(__conservation_data_path__(data_dir=self.data_dir, microphysic=mp))
            except ValueError:
                raise ValueError(f"Missing conservation dataset {mp}")
            try:
                ds = xr.open_dataset(__eulerian_data_path__(data_dir=self.data_dir, microphysic=mp))
            except ValueError:
                raise ValueError(f"Missing eulerian dataset {mp}")

        ### Merge of individual datasets

        # chunks = {'cloud_id' : 2}
        chunks = {}

        # use a list to store the datasets for each microphysics and then concatenate them afterwards
        list_of_datasets = []
        list_of_datasets_sem = []

        # iterate over all microphysics
        for mp in self.microphysics:
            print(mp)
            ds_euler = xr.open_dataset(__eulerian_data_path__(data_dir=self.data_dir, microphysic=mp))
            # use only the variables of interest
            ds_euler = ds_euler[self.variables_to_load]
            ds_euler = ds_euler.sel(time=self.time_slice)

            # reduce the surface area to a float from an array with dimension gridbox
            ds_euler["surface_area"] = ds_euler["surface_area"].mean("gridbox")
            ds_euler["surface_area"].attrs["units"] = "m^2"
            ds_euler["surface_area"].attrs["long_name"] = "Surface area"
            ds_euler["max_gridbox"] = ds_euler["max_gridbox"].fillna(ds_euler["gridbox"].max())

            # from kg per dT to g per h per m^2
            ds_conser = xr.open_dataset(
                __conservation_data_path__(data_dir=self.data_dir, microphysic=mp)
            )
            ds_conser = ds_conser.sel(time=self.time_slice)

            # get the mean and the standard error of the mean for the dataset
            ds_euler_mean, ds_euler_sem = mean_and_stderror_of_mean_dataset(
                ds=ds_euler,
                dims="time",
                keep_attrs=True,
            )
            ds_conser_mean, ds_conser_sem = mean_and_stderror_of_mean_dataset(
                ds=ds_conser,
                dims="time",
                keep_attrs=True,
            )

            # use the mean surface area

            # add the covariates to the standard error of the mean dataset
            combs = combinations(["source", "inflow", "outflow", "reservoir_change"], 2)
            for x, y in list(combs):
                var_name = f"covariance_{x}_{y}"
                ds_conser_sem[var_name] = xr.cov(ds_conser[x], ds_conser[y], dim="time")
                ds_conser_sem[var_name].attrs["long_name"] = f"Covariance between {x} and {y}"
                ds_conser_sem[var_name].attrs[
                    "units"
                ] = f"{ds_conser[x].attrs['units']} * {ds_conser[y].attrs['units']}"

            # combine the eulerian and conservation datasets
            ds_single = xr.merge([ds_conser_mean, ds_euler_mean])
            ds_single_sem = xr.merge([ds_conser_sem, ds_euler_sem])
            # expand the microphysics dimension
            ds_single = ds_single.expand_dims(microphysics=(mp,))
            ds_single_sem = ds_single_sem.expand_dims(microphysics=(mp,))

            # append the dataset to the list
            list_of_datasets.append(ds_single)
            list_of_datasets_sem.append(ds_single_sem)

        # concatenate all datasets along the microphysics dimension
        ds = xr.concat(list_of_datasets, dim="microphysics")
        ds.chunk(chunks)
        ds_sem = xr.concat(list_of_datasets_sem, dim="microphysics")
        ds_sem.chunk(chunks)

        return ds, ds_sem

    def __post_process_dataset__(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Post-process the dataset to add additional variables and convert units.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to post-process.

        Returns
        -------
        xr.Dataset
            The post-processed dataset.
        """

        ### Timestepping validation
        timestep = 2  # s
        # # extract the time step
        # timestep = ds['time'].diff().mean().values
        # timestep_sem = ds['time'].diff().std().values
        # # allowed relative tolerance for the time step
        # timestep_rtol = 0.01
        # # make sure the maximum timestep difference to the mean is less than 1%
        # if np.abs(ds['time'].diff() / timestep) > timestep_rtol :
        #     raise ValueError(f"Time step is not constant to at least {timestep_rtol * 100:.1e}%. Mean: {timestep}, std: {timestep_sem}")

        ds = __post_process_eulerian_dataset__(ds=ds)

        ds = __post_process_conservation_dataset__(
            ds=ds, da_surface_area=ds["surface_area"], timestep=timestep
        )

        return ds

    def calculate_evaporation_fraction(self):
        """
        Calculate the evaporation fraction from the dataset and the standard error of the mean dataset.
        And also for the standard error of the mean dataset for which error propagation is calculated.
        """

        A = -self.ds["source"]
        B = self.ds["inflow"]
        A_sem = self.ds_sem["source"]
        B_sem = self.ds_sem["inflow"]
        cov_AB = self.ds_sem["covariance_source_inflow"]

        f = 100 * A / B
        f.attrs["units"] = "\\%"
        f.attrs["long_name"] = "Evaporation Fraction"

        f_sem = (
            100 * (f**2) ** (0.5) * ((A_sem / A) ** 2 + (B_sem / B) ** 2 - 2 * cov_AB / (A * B)) ** (0.5)
        )
        f_sem.attrs["units"] = "\\%"
        f_sem.attrs["long_name"] = "Evaporation Fraction"

        self.ds["evaporation_fraction"] = f
        self.ds_sem["evaporation_fraction"] = f_sem

    def difference_datasets(
        self, reference_microphysics: str = "condensation"
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Calculate the difference between the dataset and a reference microphysics setup.
        The difference is calculated for the dataset and the standard error of the mean dataset.

        Parameters
        ----------
        reference_microphysics : str, optional
            The reference microphysics setup, by default "condensation".

        Returns
        -------
        Tuple[xr.Dataset, xr.Dataset]
            Tuple containing the dataset and the standard error of the mean dataset.
        """
        ds_diff = self.ds - self.ds.sel(microphysics=reference_microphysics)
        ds_diff_sem = self.ds_sem - self.ds_sem.sel(microphysics=reference_microphysics)
        return ds_diff, ds_diff_sem

    def relative_difference_datasets(
        self, reference_microphysics: str = "condensation"
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Calculate the relative difference between the dataset and a reference microphysics setup.
        The relative difference is calculated for the dataset.

        Parameters
        ----------
        reference_microphysics : str, optional
            The reference microphysics setup, by default "condensation".

        Returns
        -------
        xr.Dataset
            The relative difference dataset.
        """
        result_ds = (self.ds / self.ds.sel(microphysics=reference_microphysics) - 1) * 100
        result_ds_sem = (self.ds_sem / self.ds_sem.sel(microphysics=reference_microphysics) - 1) * 100
        return result_ds, result_ds_sem

    def normalize_gridboxes(self) -> None:
        """
        Normalize the gridboxes by the cloud base height.
        This is a mutator method.
        """

        # we need to use the data from the original dataset, as the interpolated dataset is not used for the difference calculation
        mapped_dim = self.ds["gridbox_coord3_norm"]
        new_dim = np.linspace(0, 1, 100)
        new_dim = xr.DataArray(
            new_dim,
            coords={"normalized_gridbox_coord3": new_dim},
            attrs=dict(
                long_name="Normalized gridbox coordinate",
                units="",
                description="Normalized gridbox coordinate by the maximum gridbox coordinate",
            ),
        )
        new_dim.name = "normalized_gridbox_coord3"

        ds_interpolated = interpolate_dataset(
            ds=self.ds,
            mapped_dim=mapped_dim,
            new_dim=new_dim,
            old_dim_name="gridbox",
        )
        ds_interpolated_sem = interpolate_dataset(
            ds=self.ds_sem,
            mapped_dim=mapped_dim,
            new_dim=new_dim,
            old_dim_name="gridbox",
        )
        self.ds = ds_interpolated
        self.ds_sem = ds_interpolated_sem

    def __call__(
        self,
        type: Literal["original", "difference", "relative_difference"] = "original",
    ) -> Tuple[xr.Dataset, xr.Dataset]:

        if type == "original":
            return self.ds, self.ds_sem
        elif type == "difference":
            return self.difference_datasets()
        elif type == "relative_difference":
            return self.relative_difference_datasets()
        else:
            raise ValueError(f"Unknown type {type}")
