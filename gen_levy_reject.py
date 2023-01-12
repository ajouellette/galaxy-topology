import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import mistree as mist
from Corrfunc.theory.xi import xi as calc_xi


boxsize = 75
N_samples = 500
gamma_target = -1.5
gamma_tol = 0.1
min_rvalue = 0.9


accepted = []
r0_vals = []
gamma_vals = []
iters = 0
while len(accepted) < N_samples:
    N = np.random.randint(int(5e3), int(5e4))
    lbar = boxsize / np.cbrt(N)
    t_min = 0.05 * lbar
    t_max = 1.5 * lbar

    log_t = np.random.uniform(np.log10(t_min), np.log10(t_max))
    t = 10**log_t

    print(N, lbar, t)
    sample = np.vstack(mist.get_levy_flight(N, box_size=boxsize, t_0=t, alpha=3+gamma_target)).T
    iters += 1

    zslice = np.abs(sample[:,2] - 50) < 20
    plt.plot(*sample[zslice].T[:2], '.')
    plt.show()

    print("Calculating correlation function")
    logr_max = np.log10(boxsize/8)
    logr_min = np.log10(0.75*t)
    resolution = (logr_max + 1) / 45
    xi_rbins = np.logspace(logr_min, logr_max, max(5, int((logr_max - logr_min) / resolution)))
    xi = calc_xi(boxsize, 2, xi_rbins, *sample.T, output_ravg=True)

    fit = stats.linregress(np.log(xi["ravg"]), np.log(xi["xi"]))
    gamma = fit.slope
    r0 = np.exp(-fit.intercept/gamma)
    if np.abs(gamma - gamma_target) < gamma_tol and np.abs(fit.rvalue) > min_rvalue:
        print("Accepted", r0, gamma)
        accepted.append(sample)
        r0_vals.append(r0)
        gamma_vals.append(gamma)
    else:
        print("Rejected", r0, gamma, fit.rvalue)
    print("Rejection ratio:", (iters - len(accepted)) / iters, "Accepted:", len(accepted))

    plt.plot(xi["ravg"], xi["xi"])
    plt.plot(xi_rbins, np.exp(fit.intercept + fit.slope * np.log(xi_rbins)), '--', alpha=0.7)
    plt.loglog()
    plt.show()

N_p = [len(sample) for sample in accepted]
save_name = f"levy_reject_sampling_{np.abs(gamma_target):.1f}.npz"
print("Saving to", save_name)
np.savez(save_name, data=accepted, boxsize=boxsize, lbar=boxsize/np.cbrt(N_p), N=N_p,
        r0=r0_vals, gamma=gamma_vals)
