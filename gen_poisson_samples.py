import sys
import numpy as np
from tqdm import trange


boxsize = float(sys.argv[1])

total_samples = 1000
n_samples = 50
# N = 315 to 10,000 for L=25
nbar_vals = np.logspace(-1.7, -0.2, total_samples//n_samples)
#nbar_vals = np.logspace(-2, -0.5, total_samples//n_samples)

rng = np.random.default_rng()

data = []
print("Generating data...")
for i in trange(total_samples):
    nbar = nbar_vals[i%len(nbar_vals)]
    #N = rng.poisson(nbar * boxsize**3)
    N = int(nbar * boxsize**3)
    lbar = boxsize/np.cbrt(N)

    data.append(rng.uniform(high=boxsize, size=(N, 3)))

N_vals = np.array([len(pos) for pos in data])
lbar_vals = boxsize / np.cbrt(N_vals)

save_file = f"mock_data/poisson/poisson_L{boxsize:.0f}.npz"
print("Saving to", save_file)
np.savez(save_file, data=data, boxsize=boxsize, N=N_vals, lbar=lbar_vals)
