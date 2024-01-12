# %%

import matplotlib.pyplot as plt
import numpy as np
import sdm_eurec4a.pySD.probdists as probdists
import sdm_eurec4a.pySD.rgens as rgens


# %%
rspan = [5e-9, 5e-5]  # min and max range of radii to sample [m]
radiigen = rgens.SampleLog10RadiiGen(rspan)  # radii are sampled from rspan [m]
distributions = dict()

### --- Choice of Superdroplet Radii Generator --- ###
monor = 0.05e-6  # all SDs have this same radius [m]
radiigen_monor = rgens.MonoAttrGen(monor)  # all SDs have the same radius [m]


### --- Choice of Droplet Radius Probability Distribution --- ###
geomeans = [0.02e-6, 0.2e-6, 3.5e-6]
geosigs = [1.55, 2.3, 2]
scalefacs = [0.2e6, 0.3e6, 0.025e6]
# geomeans             = [0.02e-6, 0.15e-6]
# geosigs              = [1.4, 1.6]
# scalefacs            = [6e6, 4e6]
distributions["LnNormal"] = probdists.LnNormal(geomeans, geosigs, scalefacs)

dirac0 = monor  # radius in sample closest to this value is dirac delta peak
numconc = 1e6  # total no. conc of real droplets [m^-3]
numconc = 512e6  # total no. conc of real droplets [m^-3]
distributions["DiracDelta"] = probdists.DiracDelta(dirac0)

volexpr0 = 30.531e-6  # peak of volume exponential distribution [m]
numconc = 2 ** (23)  # total no. conc of real droplets [m^-3]
distributions["VolExponential"] = probdists.VolExponential(volexpr0, rspan)

reff = 7e-6  # effective radius [m]
nueff = 0.08  # effective variance
# xiprobdist = probdists.ClouddropsHansenGamma(reff, nueff)
rdist1 = probdists.ClouddropsHansenGamma(reff, nueff)
nrain = 3000  # raindrop concentration [m^-3]
qrain = 0.9  # rainwater content [g/m^3]
dvol = 8e-4  # mean volume diameter [m]
# xiprobdist = probdists.RaindropsGeoffroyGamma(nrain, qrain, dvol)
rdist2 = probdists.RaindropsGeoffroyGamma(nrain, qrain, dvol)
numconc = 1e9  # [m^3]
distribs = [rdist1, rdist2]
scalefacs = [1000, 1]
distributions["CombinedRadiiProbDistribs"] = probdists.CombinedRadiiProbDistribs(distribs, scalefacs)

# %%

n_axs = len(distributions)
fig, axs = plt.subplots(nrows=1, ncols=n_axs, figsize=(13, 4))

if not isinstance(axs, np.ndarray):
    axs = [axs]
else:
    axs = axs.flatten()

for name, ax in zip(distributions, axs):
    N_C = 100
    radii = radiigen(nsupers=N_C)
    dist = distributions[name](radiigen(nsupers=N_C))

    ax.set_title(name)
    ax.set_xscale("log")
    ax.plot(radii, dist, marker=".", linestyle="none", label="lognormal")

# %%
