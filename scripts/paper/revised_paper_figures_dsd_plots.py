# %%
def get_mfdataset(datapath, microphysics, normheight=False):
  import glob
  import xarray as xr

  if normheight:
    ds_name = "droplet_pdfdistribs_normheight.zarr"
    
  else:
    ds_name = "droplet_pdfdistribs.zarr"
    ds_attrs = {"normalised_by_height": normheight}
  
  datasets = sorted(glob.glob(str(datapath / microphysics / "cluster_*" / "processed" / ds_name)))
  cluster_names = [Path(d).parent.parent.name for d in datasets]
  print(f"clusters found with {ds_name}:\n", cluster_names)

  ds = xr.open_mfdataset(datasets,
                           engine="zarr",
                           combine="nested",
                           concat_dim="cluster"
                           )

  def get_number_after_underscore(s):
    parts = s.split("_")
    return int(parts[-1]) if len(parts) > 1 else None

  cluster_nums = dict(cluster=("cluster", [get_number_after_underscore(c) for c in cluster_names]))
  ds = ds.assign_coords(cluster_nums)
  
  ds.attrs = {"normalised_by_height": normheight}

  return ds

# %%
from pathlib import Path
datapath = Path("/Users/yoctoyotta1024/Downloads/rain-evap-nils/sdm-eurec4a-CLEO/data/output_v4.2")
microphysics = "condensation"
ds = get_mfdataset(datapath, microphysics, normheight=True)
ds.sel(cluster=81)
# %%
# %% ### TODO:
# - bash script and documentation for dsd dataset creation
# - select particular clusters to plot
# --> goal: plot for many clusters and for diff microphysics
