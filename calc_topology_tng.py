import sys
import os
import numpy as np
from scipy.interpolate import interp1d
import gudhi as gd
import gudhi.representations as gdr
from alpha_complex_periodic import calc_persistence
import illustris_python as tng


# selection cuts
n_star_cut = 50
mass_cut = 3e8
ssfr_cut = 10**-10.5
match_ssfr = True

# possible selections
selections = ["galaxy", "halo", "sfgal", "qsntgal"]


def main():

    if len(sys.argv) < 2:
        print("Must provide simulation name and object selection")
        sys.exit(1)
    sim = sys.argv[1]
    selection = sys.argv[2]

    if selection not in selections:
        print("selection must be one of", selections)
        sys.exit(1)

    # z=0 snapshot
    if "TNG" in sim:
        snap_num = 99
    elif "Illustris" in sim:
        snap_num = 135
    else:
        print("Unknown simulation")
        sys.exit()

    subhalo_fields = ["SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloSFR"] \
            + ["SubhaloFlag"] if "Illustris" not in sim else []
    group_fields = ["GroupPos", "Group_M_Crit200"]

    header = tng.groupcat.loadHeader(sim, snap_num)
    boxsize = header["BoxSize"] / 1e3

    alpha_resolution = int(boxsize/2 * 50)
    alpha_range = [0, boxsize/2]
    alpha = np.linspace(*alpha_range, alpha_resolution)
    scaled_resolution = 500
    scaled_range = [0, 5]
    alpha_scaled = np.linspace(*scaled_range, scaled_resolution)

    # Topological summaries
    DS = gdr.preprocessing.DiagramSelector(use=True)
    ES = gdr.Entropy(mode="vector", resolution=alpha_resolution, sample_range=alpha_range,
            normalized=False)

    print("Loading simulation data")

    subhalo_data = tng.groupcat.loadSubhalos(sim, snap_num, fields=subhalo_fields)
    group_data = tng.groupcat.loadGroups(sim, snap_num, fields=group_fields)

    pos = data["SubhaloPos"] / 1e3
    dm_mass = data["SubhaloMassType"][:,1] * 1e10
    st_mass = data["SubhaloMassType"][:,4] * 1e10
    tot_mass = np.sum(data["SubhaloMassType"], axis=1) * 1e10
    if "Illustris" not in sim:
        flag = data["SubhaloFlag"].astype(bool)
    else:
        flag = dm_mass / tot_mass > 0.1
    n_stars = data["SubhaloLenType"][:,4]

    print(f"min n_star with mass cut: {np.min(n_stars[(st_mass > mass_cut) * flag])}")
    print(f"min mass with n_star cut: {np.min(st_mass[(n_stars > n_star_cut) * flag]):.3e}")

    galaxy = (n_stars > n_star_cut) * (st_mass > mass_cut) * flag
    ssfr = data["SubhaloSFR"][galaxy] / st_mass[galaxy]

    # use abundance matching to get halos
    halo = np.isin(np.arange(len(pos)), np.argsort(dm_mass)[::-1][:np.sum(galaxy)])

    if not match_ssfr:
        sf = ssfr > ssfr_cut
        qsnt = ssfr <= ssfr_cut
    else:
        qsnt = np.argsort(ssfr)[:len(ssfr)//2]
        sf = np.argsort(ssfr)[len(ssfr)//2:]

    print(f"sSFR cut: {np.max(ssfr[qsnt]):.3e}")

    if selection == "galaxy":
        pos = pos[galaxy]
    elif selection == "halo":
        pos = pos[halo]
    elif selection == "sfgal":
        pos = pos[galaxy][sf]
    elif selection == "qsntgal":
        pos = pos[galaxy][qsnt]

    n_selection = len(pos)
    print(n_selection, "objects selected")
    lbar = boxsize / np.cbrt(n_selection)

    pairs = calc_persistence(pos, boxsize=boxsize, precision="fast")
    pairs = [np.array(pairs[d]) for d in range(3)]

    print("Calculating ES curves")
    es = ES.fit_transform(DS.fit_transform(pairs))

    print("Scaling and interpolating ES curves")
    interp = interp1d(alpha/lbar, es, axis=-1, bounds_error=False, fill_value=0, assume_sorted=True, kind="slinear")
    es_scaled = interp(alpha_scaled)
    norm = np.trapz(np.abs(es_scaled), alpha_scaled, axis=-1)
    es_scaled /= np.expand_dims(norm, -1)

    sim_name = sim.split('/')[-1]
    fname = f"topology_summaries/{sim_name}/es_{selection}.npz"
    print(f"Saving data to {fname}")
    np.savez(fname, alpha=alpha, es=es, alpha_scaled=alpha_scaled, es_scaled=es_scaled, n_selection=n_selection, lbar=lbar)


main()
