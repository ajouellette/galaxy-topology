import sys
import glob
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import illustris_python as tng


def calc_hmf(bins, masses, boxsize):
    hmf = np.zeros_like(bins)
    for i in range(len(bins)):
        masses = masses[masses > bins[i]]
        hmf[i] = len(masses) / boxsize**3
    return hmf


if __name__ == "__main__":
    bins = np.logspace(9, 15, 25)

    print("Rockstar catalogs...")
    t_start = time.perf_counter()
    catalogs = glob.glob("camels-sam/CV_*/rockstar_99.hdf5")
    hmfs_rockstar = []
    for catalog in catalogs:
        with h5py.File(catalog) as f:
            masses = f["M200c"][:]

        hmf = calc_hmf(bins, masses, 100)
        hmfs_rockstar.append(hmf)

    hmfs_rockstar = np.vstack(hmfs_rockstar)
    t_end = time.perf_counter()
    print(f"Done. {t_end-t_start:.2f} s")

    tng_fields = ["Group_M_Crit200"]
    tng100 = "TNG100"
    tng300 = "TNG300"
    illustris = "Illustris"

    print("TNG100..")
    t_start = time.perf_counter()
    boxsize = tng.groupcat.loadHeader(tng100, 99)["BoxSize"] / 1e3
    masses = tng.groupcat.loadHalos(tng100, 99, fields=tng_fields) * 1e10
    hmf_tng100 = calc_hmf(bins, masses, boxsize)
    t_end = time.perf_counter()
    print(f"Done. {t_end-t_start:.2f} s")

    print("TNG300..")
    t_start = time.perf_counter()
    boxsize = tng.groupcat.loadHeader(tng300, 99)["BoxSize"] / 1e3
    masses = tng.groupcat.loadHalos(tng300, 99, fields=tng_fields) * 1e10
    hmf_tng300 = calc_hmf(bins, masses, boxsize)
    t_end = time.perf_counter()
    print(f"Done. {t_end-t_start:.2f} s")

    print("Illustris..")
    t_start = time.perf_counter()
    boxsize = tng.groupcat.loadHeader(illustris, 135)["BoxSize"] / 1e3
    masses = tng.groupcat.loadHalos(illustris, 135, fields=tng_fields) * 1e10
    hmf_illustris = calc_hmf(bins, masses, boxsize)
    t_end = time.perf_counter()
    print(f"Done. {t_end-t_start:.2f} s")

    med = np.mean(hmfs_rockstar, axis=0)
    lq = np.quantile(hmfs_rockstar, 0.025, axis=0)
    hq = np.quantile(hmfs_rockstar, 0.975, axis=0)

    plt.plot(bins, med)
    plt.fill_between(bins, lq, hq, alpha=0.2, label="CAMELS/rockstar")
    plt.plot(bins, hmf_tng300, 'o', label="TNG300/subfind")
    plt.plot(bins, hmf_tng100, 'o', label="TNG100/subfind")
    plt.plot(bins, hmf_illustris, 'x', label="Illustris/subfind")
    plt.legend()
    plt.loglog()
    plt.xlabel("$M_{200c}$ [$h^{-1}$ M$_{\\odot}$]")
    plt.ylabel("$n(\,>M)$ [$h^3$ Mpc$^{-3}$]")
    plt.show()
