import glob
import sys

import numpy as np


suite = sys.argv[1]
sim_set = sys.argv[2]

sims = glob.glob(suite + "/" + sim_set + "_*")
print(f"{len(sims)} sims found")

params = np.zeros((len(sims), 6))
for sim in sims:
    i = int(sim.split("_")[-1])
    params[i] = np.loadtxt(sim + "/CosmoAstro_params.txt")

suite_name = suite.split('/')[-1]
np.save(f"CosmoAstro_params_{suite_name}_{sim_set}", params)
