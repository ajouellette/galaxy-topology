import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import powerbox as pbox
from Corrfunc.theory.xi import xi as calc_xi


N_grid = 256
boxsize = 100
N_samples = 500
gamma_target = -1.2
gamma_tol = 0.1
N = int(1e4)

A_min = 0.5 * boxsize**3 / N * (np.pi * N_grid/boxsize)**(3+gamma_target)
A_max = 75

xi_rbins = np.logspace(np.log10(0.5*boxsize/N_grid), np.log10(boxsize/8), 45)

accepted = []
r0_vals = []
gamma_vals = []
iters = 0
while len(accepted) < N_samples:
    logA = np.random.uniform(np.log10(A_min), np.log10(A_max))
    A = 10**logA
    #A = np.random.uniform(A_min, A_max)
    pbl = pbox.LogNormalPowerBox(N_grid, dim=3, pk = lambda k: A/k**(3+gamma_target), boxlength=boxsize)
    sample = pbl.create_discrete_sample(N / boxsize**3, min_at_zero=True)
    iters += 1

    #zslice = np.abs(sample[:,2] - 50) < 20
    #plt.plot(*sample[zslice].T[:2], '.')
    #plt.show()

    print("Calculating correlation function")
    xi = calc_xi(boxsize, 2, xi_rbins, *sample.T, output_ravg=True)
    fit = stats.linregress(np.log(xi["ravg"]), np.log(xi["xi"]))
    gamma = fit.slope
    r0 = np.exp(-fit.intercept/gamma)
    if np.abs(gamma - gamma_target) < gamma_tol and np.abs(fit.rvalue) > 0.99:
        print("Accepted", r0, gamma)
        accepted.append(sample)
        r0_vals.append(r0)
        gamma_vals.append(gamma)
    else:
        print("Rejected", r0, gamma, "rvalue:", fit.rvalue)
    print("Rejection ratio:", (iters - len(accepted)) / iters, "Accepted:", len(accepted))

    #plt.plot(xi["ravg"], xi["xi"], label="sample")
    #plt.plot(xi_rbins, np.exp(fit.intercept + fit.slope * np.log(xi_rbins)), '--', alpha=0.7, label="input")
    #plt.loglog()
    #plt.legend()
    #plt.show()

N_p = [len(sample) for sample in accepted]
save_name = f"lognormal_reject_sampling_{np.abs(gamma_target):.1f}.npz"
print("Saving to", save_name)
np.savez(save_name, data=accepted, boxsize=boxsize, r0=r0_vals, gamma=gamma_vals,
        N=N_p, lbar=boxsize/np.cbrt(N_p))
