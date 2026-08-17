# %% Thread-safe print wrapper for use of script with ThreadPoolExecutor
import threading
_print_lock = threading.Lock()
def tprint(*args, **kwargs):
    """Thread-safe print wrapper that preserves print kwargs."""
    with _print_lock:
        print(*args, **kwargs)

# %%
### ------------------------- FUNCTION DEFINITIONS ------------------------- ###
def parse_arguments():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path2sdmeurec4aCLEO", type=Path, help="Absolute path to sdmeurec4aCLEO directory"
    )
    parser.add_argument(
        "path2zarr", type=Path, help="Absolute path to .zarr Cleo dataset"
    )
    parser.add_argument(
        "path2setup", type=Path, help="Absolute path to setup.txt file"
    )
    parser.add_argument(
        "path2grid", type=Path, help="Absolute path to gridbox binary file"
    )
    parser.add_argument(
        "path2output", type=Path, help="Directory to write datasets in",
    )
    return parser.parse_args()

def check_paths(path2sdmeurec4aCLEO, path2zarr, path2setup, path2grid, path2output):
    assert (
            path2sdmeurec4aCLEO
        ).is_dir(), "sdmeurec4aCleo directory not found"

    assert (
            path2zarr
        ).is_dir(), "Cleo .zarr directory for input doesn't exist"

    assert (
            path2setup
        ).is_file(), "setupfile for input doesn't exist"

    assert (
            path2grid
        ).is_file(), "gridfile for input doesn't exist"

    assert (
            path2output
        ).parent.is_dir(), "Directory for output doesn't exist"

    tprint("path2sdmeurec4aCLEO:", path2sdmeurec4aCLEO)
    tprint("path2zarr:", path2zarr)
    tprint("path2setup:", path2setup)
    tprint("path2grid:", path2grid)
    tprint("path2output:", path2output)

def get_cluster_name(path2zarr):
    return str(path2zarr.parent.name)  # "cluster_XXX"

def get_config_consts_gridboxes(path2sdmeurec4aCLEO, path2setup, path2grid):
    import sys
    
    sys.path.append(str(path2sdmeurec4aCLEO))
    from pySD.sdmout_src import pysetuptxt, pygbxsdat

    config = pysetuptxt.get_config(path2setup, nattrs=3, isprint=False)
    consts = pysetuptxt.get_consts(path2setup, isprint=False)
    gbxs = pygbxsdat.get_gridboxes(path2grid, consts["COORD0"], isprint=False)
    return config, consts, gbxs


def reshape_superdroplet_attribute(ds, var, start_time, end_time):
    import awkward as ak
    import numpy as np

    raggedcount = ds.raggedcount.values
    nsupers = ds.nsupers.values
    data = ak.unflatten(ds[var].values, raggedcount)
    data = ak.unflatten(data, ak.flatten(nsupers), axis=1)

    idx1 = np.argmin(abs(start_time - ds.time.values))
    idx2 = np.argmin(abs(end_time - ds.time.values)) + 1
    data = ak.to_regular(data[idx1:idx2], axis=1)

    assert (data.layout.size == ds.sizes['gbxindex']), 'ragged array not shaped correctly per gridbox'

    return data

def distribution_in_log10radius_bins(nbins, rspan, radius, nsamples, weights, per_bin=True):
    import numpy as np

    assert np.all(radius < rspan[1]), "maximum radius bin too small for largest droplets in Cleo dataset"

    hedges = np.linspace(np.log10(rspan[0]), np.log10(rspan[1]), nbins + 1)  # edges to log10 bins
    hist, hedges = np.histogram(np.log10(radius), bins=hedges, weights=weights, density=None)
    hcens = (np.power(hedges[1:], 10) + np.power(hedges[:-1], 10)) / 2  # lnr bin centres

    hcens = (hedges[1:] + hedges[:-1]) / 2  # log10 bin centres

    hist = hist / nsamples

    edges = np.power(10, hedges)  # [microns]
    centers = np.power(10, hcens)  # [microns]
    widths = (edges[1:] - edges[:-1]) / 1e6   # linear radius widths [m]

    if per_bin:
        hist = hist / widths

    return hist, edges, centers, widths

def nsupers_weighting(consts, radius, xi, vol):
    return None # [nsupers]

def numconc_weighting(consts, radius, xi, vol):
    return xi / vol / 1e6 # [droplets / cm^3]

def massconc_weighting(consts, radius, xi, vol):
    import numpy as np
    r = radius * 1e-6  # [m]
    rho = consts["RHO_L"] * 1000  # [g/m^3]
    mass = 4.0/3.0 * np.pi * r * r * r * rho
    return xi * mass / vol  # real droplets [g/m^3]

def calculate_pdf_distributions(path2sdmeurec4aCLEO, path2zarr, path2setup, path2grid):
    import awkward as ak
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(path2zarr, engine="zarr")

    nbins = 100
    rspan = [1e-3, 5e4] # [min, max] radius of distribution
    start_time = 1500 # [s]
    end_time = ds.time.values[-1] # [s]

    radius = reshape_superdroplet_attribute(ds, "radius", start_time, end_time)
    xi = reshape_superdroplet_attribute(ds, "xi", start_time, end_time)

    config, consts, gbxs = get_config_consts_gridboxes(path2sdmeurec4aCLEO, path2setup, path2grid)
    nsupers_pdf = []
    numconc_pdf = []
    massconc_pdf = []
    for i in range(ds.sizes['gbxindex']):
        nsamples = len(radius[:,i]) # number of timesteps to average over
        r = ak.flatten(radius[:,i])
        x = ak.flatten(xi[:,i])
        v = gbxs["gbxvols"][0,0,i]

        d, edgs, cens, wdths = distribution_in_log10radius_bins(nbins, rspan, r,  nsamples,
                                                                nsupers_weighting(consts, r, x, v))
        nsupers_pdf.append(d)

        d = distribution_in_log10radius_bins(nbins, rspan, r,  nsamples,
                                            numconc_weighting(consts, r, x, v))[0]
        numconc_pdf.append(d)

        d = distribution_in_log10radius_bins(nbins, rspan, r,  nsamples,
                                            massconc_weighting(consts, r, x, v))[0]
        massconc_pdf.append(d)

    return nsupers_pdf, numconc_pdf, massconc_pdf, edgs, cens, wdths, gbxs

def create_pdf_distribution_dataset(path2sdmeurec4aCLEO, path2zarr, path2setup, path2grid):
    import numpy as np
    import xarray as xr

    nsupers_pdf, numconc_pdf, massconc_pdf, edgs, cens, wdths, gbxs = calculate_pdf_distributions(path2sdmeurec4aCLEO, path2zarr, path2setup, path2grid)

    nsupers_pdf = xr.DataArray(
        np.asarray(nsupers_pdf),
        name="nsupers_pdf",
        dims=("height", "centers"),
        attrs={
            "long_name": "number of superdroplets per log-radius bin",
            "units": "m^-1"
        },
    )

    numconc_pdf = xr.DataArray(
        np.asarray(numconc_pdf),
        name="numconc_pdf",
        dims=("height", "centers"),
        attrs={
            "long_name": "droplet number concentration per log-radius bin",
            "units": "cm^-3 m^-1"
        },
    )

    massconc_pdf = xr.DataArray(
        np.asarray(massconc_pdf),
        name="massconc_pdf",
        dims=("height", "centers"),
        attrs={
            "long_name": "droplet mass concentration per log-radius bin",
            "units": "g m^-3 m^-1"
        },
    )

    edges = xr.DataArray(
        np.asarray(edgs),
        name="edges",
        dims=("edges"),
        attrs={
            "long_name": "edges of bins evenly spaced in log-radius",
            "units": "µm"
        },
    )

    centers = xr.DataArray(
        np.asarray(cens),
        name="centers",
        dims=("centers"),
        attrs={
            "long_name": "centers of bins evenly spaced in log-radius",
            "units": "µm"
        },
    )

    widths = xr.DataArray(
        np.asarray(wdths),
        name="widths",
        dims=("widths"),
        attrs={
            "long_name": "radius bin widths(evenly spaced in log-radius)",
            "units": "m"
        },
    )

    height = xr.DataArray(
            gbxs["zfull"],
            name="height",
            dims=("height"),
            attrs={
                "long_name": "z coordinate centers of gridbox cells",
                "units": "m"
            },
        )

    height_2 = xr.DataArray(
            gbxs["zhalf"],
            name="height_2",
            dims=("height_2"),
            attrs={
                "long_name": "z coordinate edges of gridbox cells",
                "units": "m"
            },
        )

    ds = xr.Dataset(
    {
        "height": height,
        "height_2": height_2,
        "centers": centers,
        "edges": edges,
        "widths": widths,
        "nsupers_pdf": nsupers_pdf,
        "numconc_pdf": numconc_pdf,
        "massconc_pdf": massconc_pdf,
    }
    )

    return ds

def normalise_dataset_heights(ds, nlevels=50):
    import numpy as np

    # normalise height levels between 0 and 1.0
    level = (ds.height - ds.height.min()) / (ds.height.max() - ds.height.min())

    # replace height dim with level
    ds_norm = ds.copy(deep=True)
    ds_norm = ds_norm.rename_dims({"height": "level"})
    for v in ["height", "height_2"]:
        ds_norm = ds_norm.drop_vars(v)
    ds_norm = ds_norm.assign_coords(level=("level", level.values))
    
    # interpolate to 50 normalized height levels between 0.0 and 1.0
    new_level = np.linspace(0.0, 1.0, nlevels)
    ds_norm = ds_norm.interp(level=new_level)
    ds_norm = ds_norm.assign_coords(level=new_level)

    # provide level metadata
    ds_norm.coords["level"].attrs["units"] = None
    ds_norm.coords["level"].attrs["long_name"] = "normalised height level between 0.0 and 1.0"
    
    return ds_norm

def write_zarr_dataset(ds, ds_name):
    if ds_name.is_dir():
        tprint(f"WARNING: not writing {ds_name}, dataset already exists")
    else:
        ds.to_zarr(ds_name, mode="w")
        tprint(f"written dataset: {ds_name}")

# %% sanity check plotting functions
def sanity_check_plots(ds, ds_norm):
    import matplotlib.pyplot as plt
    import numpy as np

    ### ------------------------ ###
    MINRADIUS = 5e-05                                                   # minimum radius of new super-droplets [m]
    MAXRADIUS = 0.003                                                   # maximum radius of new super-droplets [m]
    NUMCONC_a = 83677622.51847906                                       # number conc. of 1st droplet lognormal dist [m^-3]
    GEOMEAN_a = 9.666635869537405e-06                                   # geometric mean radius of 1st lognormal dist [m]
    geosigma_a = 1.6251520387071434                                     # geometric standard deviation of 1st lognormal dist
    NUMCONC_b = 21.081727728701953                                      # number conc. of 2nd droplet lognormal dist [m^-3]
    GEOMEAN_b = 0.00012860636070712326                                  # geometric mean radius of 2nd lognormal dist [m]
    geosigma_b = 1.538549427061947                                      # geometric standard deviation of 2nd lognormal dist

    # Create radius array
    r = np.logspace(np.log10(MINRADIUS), np.log10(MAXRADIUS), 100)

    # Lognormal distribution: pdf = (numconc / (sqrt(2*pi) * ln(sigma) * r)) * exp(-0.5 * (ln(r/geomean))^2 / ln(sigma)^2)
    ln_sigma_a = np.log(geosigma_a)
    ln_sigma_b = np.log(geosigma_b)

    pdf_a = (NUMCONC_a / (np.sqrt(2 * np.pi) * ln_sigma_a * r)) * np.exp(-0.5 * (np.log(r / GEOMEAN_a) / ln_sigma_a) ** 2)
    pdf_b = (NUMCONC_b / (np.sqrt(2 * np.pi) * ln_sigma_b * r)) * np.exp(-0.5 * (np.log(r / GEOMEAN_b) / ln_sigma_b) ** 2)

    pdf_total = pdf_a + pdf_b

    plt.figure(figsize=(10, 6))
    plt.loglog(r*1e6, pdf_total/1e6, label='Sum', linewidth=2)
    plt.xlabel('Radius [microns]')
    plt.ylabel('Number concentration per bin [cm^-3 m^-1]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Compare to Sum of two lognormal distributions')
    plt.xlim(10, 1e4)
    plt.ylim(1-10, 1e4)

    plt.plot(ds.edges[:-1], ds.numconc_pdf.sel(height=ds.height.max()))

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(10, 1e4)
    plt.ylim(1-10, 1e4)

    plt.show()

    ### ------------------------ ###
    for dist in ds.numconc_pdf:
        plt.step(ds.edges[:-1], dist, where="pre")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("All gridboxes' numconc pdfs")
    plt.ylabel('Number concentration per bin [cm^-3 m^-1]')
    plt.xlabel('Radius [microns]')

    plt.show()

    ### ------------------------ ###
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(8, 12))
    fig.suptitle("Distributions at Certain Height")

    heights2plot = [710, 510, 310, 110, 10]

    for i, h in enumerate(heights2plot):
        axs = axes[i,:]
        axs[0].text(1.5e-1, 4e1, f"{h:.0f}m")

        axs[0].step(ds.centers, ds.widths.values*ds.nsupers_pdf.sel(height=h, method="nearest"), where="mid")
        axs[1].step(ds.centers, ds.numconc_pdf.sel(height=h, method="nearest"), where="mid")
        axs[2].step(ds.centers, ds.massconc_pdf.sel(height=h, method="nearest"), where="mid")


    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1e-1, 5e4])
    for ax in axes[-1,:]:
        ax.set_xlabel("radius / microns")
    axes[0,0].set_title("nsupers")
    axes[0,1].set_title("numconc / cm^-3 m^-1")
    axes[0,2].set_title("nsupers / g m^-3 m^-1")

    plt.show()

    ### ------------------------ ###
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(8, 12))
    fig.suptitle("Distributions at Certain Levels")

    levels2plot = np.asarray([(h - ds.height.min()) / (ds.height.max() - ds.height.min()) for h in heights2plot])

    for i, l in enumerate(levels2plot):
        axs = axes[i,:]
        axs[0].text(1.5e-1, 4e1, f"level={l:.2f}")

        axs[0].step(ds_norm.centers, ds_norm.widths.values*ds_norm.nsupers_pdf.sel(level=l, method="nearest"), where="mid")
        axs[1].step(ds_norm.centers, ds_norm.numconc_pdf.sel(level=l, method="nearest"), where="mid")
        axs[2].step(ds_norm.centers, ds_norm.massconc_pdf.sel(level=l, method="nearest"), where="mid")

    for ax in axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1e-1, 5e4])
    for ax in axes[-1,:]:
        ax.set_xlabel("radius / microns")
    axes[0,0].set_title("nsupers")
    axes[0,1].set_title("numconc / cm^-3 m^-1")
    axes[0,2].set_title("nsupers / g m^-3 m^-1")

    plt.show()

# %% ### ------------------------- MAIN PROGRAM ------------------------- ### 
def main(path2sdmeurec4aCLEO, path2zarr, path2setup, path2grid, path2output, plot_sanitycheck=False):
    import xarray as xr

    check_paths(path2sdmeurec4aCLEO, path2zarr, path2setup, path2grid, path2output)

    ds_name = path2output / "droplet_pdfdistribs.zarr"
    ds_norm_name = path2output / "droplet_pdfdistribs_normheight.zarr"

    if ds_name.is_dir() and ds_norm_name.is_dir(): 
        tprint("opening existing datasets")
        ds = xr.open_dataset(ds_name, engine="zarr")
        ds_norm = xr.open_dataset(ds_norm_name, engine="zarr")
    else:
        tprint("writing new datasets")
        ds = create_pdf_distribution_dataset(path2sdmeurec4aCLEO, path2zarr, path2setup, path2grid)
        ds_norm = normalise_dataset_heights(ds)

        ds.attrs = {"cluster": get_cluster_name(path2zarr)}
        ds_norm.attrs = {"cluster": get_cluster_name(path2zarr)}

        write_zarr_dataset(ds, ds_name)
        write_zarr_dataset(ds_norm, ds_norm_name)

    if plot_sanitycheck:
        sanity_check_plots(ds, ds_norm)

    return ds, ds_norm

# %% Run main
if __name__ == "__main__":
    args = parse_arguments()
    main(
        args.path2sdmeurec4aCLEO,
        args.path2zarr,
        args.path2setup,
        args.path2grid,
        args.path2output,
        plot_sanitycheck=False,
    )

# # %% EXAMPLE FOR TESTING
# from pathlib import Path

# rootpath = Path("/Users/yoctoyotta1024/Downloads/rain-evap-nils/sdm-eurec4a-CLEO/data/output_v4.2/condensation/cluster_81")
# path2sdmeurec4aCLEO = Path("/Users/yoctoyotta1024/Documents/c1_springsummer2024/nils_masters/rain-evap-nils/sdm-eurec4a-CLEO/")
# path2zarr = rootpath / "eurec4a1d_sol.zarr"
# path2setup = rootpath / "config" / "eurec4a1d_setup.txt"
# path2grid = rootpath / "share" / "eurec4a1d_ddimlessGBxboundaries.dat"
# path2output = rootpath / "processed"

# main(
#     path2sdmeurec4aCLEO,
#     path2zarr,
#     path2setup,
#     path2grid,
#     path2output,
#     plot_sanitycheck=True,
# )
