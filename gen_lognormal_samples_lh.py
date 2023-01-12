import sys
import numpy as np
from scipy.stats.qmc import LatinHypercube, scale
from tqdm import trange
from mpi4py import MPI
import powerbox as pb


boxsize = float(sys.argv[1])

# log-normal fields with power-law power spectrum
# P(k) = A/k**n
# 3 parameters: nbar, A, n

total_samples = 500
print("Generating sample parameters...")
sampler = LatinHypercube(d=3)
params = sampler.random(total_samples)

log_nbar_range = [-1.7, -0.2]
log_A_range = [1, 2.4]
n_range = [1.2, 1.8]

log_nbar, log_A, n_vals = scale(params, [log_nbar_range[0], log_A_range[0], n_range[0]],
                                        [log_nbar_range[1], log_A_range[1], n_range[1]]).T

nbar_vals = 10**log_nbar
A_vals = 10**log_A

N_grid = 256

data = []
print("Generating data...")
for i in trange(total_samples):
    #if not (i*100 // total_samples % 10):
    #    print(f"{100*i//total_samples:.0f}%")
    nbar = nbar_vals[i]
    A = A_vals[i]
    n = n_vals[i]

    pbl = pb.LogNormalPowerBox(N_grid, dim=3, pk=lambda k: A/k**n, boxlength=boxsize)
    data.append(pbl.create_discrete_sample(nbar, min_at_zero=True))
    del pbl

N_vals = np.array([len(pos) for pos in data])
lbar_vals = boxsize / np.cbrt(N_vals)

save_file = f"mock_data/lognormal/lognormal_L{boxsize:.0f}.npz"
print("Saving to", save_file)
np.savez(save_file, data=data, boxsize=boxsize, N=N_vals, lbar=lbar_vals, A=A_vals, n=n_vals)
