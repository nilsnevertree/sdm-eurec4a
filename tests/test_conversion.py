# %%
import numpy as np
import xarray as xr

from sdm_eurec4a.conversions import lwc_from_psd, msd_from_psd, vsd_from_psd


radii = np.array([1, 2, 3])

psd = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ]
)
ds_example_si = xr.Dataset(
    {
        "particle_size_distribution": (
            ["time", "radius"],
            psd,
            {"unit": "#/m^3"},
        ),
        "radius": ("radius", radii, dict(unit="m")),
    },
    coords={
        "time": np.array([0, 1]),
        "diameter": ("radius", radii * 2, dict(unit="m")),
    },
)

ds_example_non_si = xr.Dataset(
    {
        "particle_size_distribution": (
            ["time", "radius"],
            1e-6 * psd,
            {"unit": "#/dm^3"},
        ),
        "radius": ("radius", 1e-6 * radii, dict(unit="µm")),
    },
    coords={
        "time": np.array([0, 1]),
        "diameter": ("radius", 1e-6 * radii * 2, dict(unit="µm")),
    },
)


# The true results are calculated with the following formulas:
# VSD = PSD * 4/3 * pi * radius^3
# MSD = VSD * density
radii2D = np.array([radii, radii])
water_density = 1e3
volume = 4 / 3 * np.pi * radii2D**3 * psd
mass = volume * water_density
lwc = np.sum(mass, axis=1)

ds_should = xr.Dataset(
    {
        "volume_size_distribution": (
            ["time", "radius"],
            volume,
            {
                "units": "m^3/m^3",
                "long_name": "Volume size distribution",
                "description": "Volume size distribution calculated from the particle size distribution. VSD = PSD * 4/3 * pi * radius^3",
            },
        ),
        "mass_size_distribution": (
            ["time", "radius"],
            mass,
            {
                "units": "kg/m^3",
                "long_name": "Mass size distribution",
                "description": "Mass size distribution calculated from the particle size distribution. MSD = rho_water * PSD * 4/3 * pi * radius^3",
            },
        ),
        "liquid_water_content": (
            ["time"],
            lwc,
            {
                "units": "kg/m^3",
                "long_name": "Liquid water content",
                "description": "Liquid water content calculated from the particle size distribution. LWC is the sum over specified dimension of (rho_water * PSD * 4/3 * pi * radius^3)",
            },
        ),
    },
    coords={
        "time": np.array([0, 1]),
        "radius": radii,
        "diameter": ("radius", radii * 2, dict(unit="m")),
    },
)


def test_vsd_from_psd():
    """
    Test the function vsd_from_psd For realtive accuracy of.

    1.5 * 10**(-decimal)

    Handles the following cases:
    - SI units
    - non-SI units
    - diameter instead of radius
    """
    # Accuracy of the test
    decimal = 10
    rtol = 1.5 * 10 ** (-decimal)

    # SI UNITS
    vsd = vsd_from_psd(ds=ds_example_si)
    # Also choords should be the same
    xr.testing.assert_allclose(vsd, ds_should["volume_size_distribution"], rtol=rtol)
    assert vsd.name == "volume_size_distribution"
    assert vsd.attrs == ds_should["volume_size_distribution"].attrs
    # SI UNITS diameter
    vsd = vsd_from_psd(ds=ds_example_si, scale_name="diameter", radius_given=False)
    xr.testing.assert_allclose(vsd, ds_should["volume_size_distribution"], rtol=rtol)
    assert vsd.name == "volume_size_distribution"
    assert vsd.attrs == ds_should["volume_size_distribution"].attrs
    # non-SI UNITS
    vsd = vsd_from_psd(ds=ds_example_non_si, psd_factor=1e6, scale_factor=1e6)
    # the coords will not be equal, so just look at the data
    np.testing.assert_array_almost_equal(
        vsd.data, ds_should["volume_size_distribution"].data, decimal=decimal
    )
    assert vsd.name == "volume_size_distribution"
    assert vsd.attrs == ds_should["volume_size_distribution"].attrs


def test_msd_from_psd():
    """
    Test the function msd_from_psd For relative accuracy of.

    1.5 * 10**(-decimal)

    Handles the following cases:
    - SI units
    - non-SI units
    - diameter instead of radius
    """
    # Accuracy of the test
    decimal = 10
    rtol = 1.5 * 10 ** (-decimal)

    # SI UNITS
    msd = msd_from_psd(ds=ds_example_si)
    # Also coords should be the same
    xr.testing.assert_allclose(msd, ds_should["mass_size_distribution"], rtol=rtol)
    assert msd.name == "mass_size_distribution"
    assert msd.attrs == ds_should["mass_size_distribution"].attrs
    # SI UNITS diameter
    msd = msd_from_psd(ds=ds_example_si, scale_name="diameter", radius_given=False)
    xr.testing.assert_allclose(msd, ds_should["mass_size_distribution"], rtol=rtol)
    assert msd.name == "mass_size_distribution"
    assert msd.attrs == ds_should["mass_size_distribution"].attrs
    # non-SI UNITS
    msd = msd_from_psd(ds=ds_example_non_si, psd_factor=1e6, scale_factor=1e6)
    # the coords will not be equal, so just look at the data
    np.testing.assert_array_almost_equal(
        msd.data, ds_should["mass_size_distribution"].data, decimal=decimal
    )
    assert msd.name == "mass_size_distribution"
    assert msd.attrs == ds_should["mass_size_distribution"].attrs


def test_lwc_from_psd():
    """
    Test the function lwc_from_psd For relative accuracy of.

    1.5 * 10**(-decimal)

    Handles the following cases:
    - SI units
    - non-SI units
    - diameter instead of radius
    """
    # Accuracy of the test
    decimal = 12
    rtol = 1.5 * 10 ** (-decimal)

    # SI UNITS
    lwc = lwc_from_psd(ds=ds_example_si)
    # Also coords should be the same
    xr.testing.assert_allclose(lwc, ds_should["liquid_water_content"], rtol=rtol)
    assert lwc.name == "liquid_water_content"
    assert lwc.attrs == ds_should["liquid_water_content"].attrs
    # SI UNITS diameter
    lwc = lwc_from_psd(ds=ds_example_si, scale_name="diameter", radius_given=False)
    xr.testing.assert_allclose(lwc, ds_should["liquid_water_content"], rtol=rtol)
    assert lwc.name == "liquid_water_content"
    assert lwc.attrs == ds_should["liquid_water_content"].attrs
    # non-SI UNITS
    lwc = lwc_from_psd(ds=ds_example_non_si, psd_factor=1e6, scale_factor=1e6)
    # the coords will not be equal, so just look at the data
    np.testing.assert_array_almost_equal(
        lwc.data, ds_should["liquid_water_content"].data, decimal=decimal
    )
    assert lwc.name == "liquid_water_content"
    assert lwc.attrs == ds_should["liquid_water_content"].attrs
