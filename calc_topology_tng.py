import sys
import os
import numpy as np
from scipy.interpolate import interp1d
import gudhi as gd
import gudhi.representations as gdr
from Corrfunc.theory.xi import xi
import powerbox as pb
from alpha_complex_periodic import calc_persistence
import illustris_python as tng


# selection cuts
gal_mass_cut = 2e8
halo_mass_cut = 1e10
ssfr_cut = 10**-10.5
match_ssfr = True

# possible selections
selections = ["galaxy", "halo", "sfgal", "qsntgal"]


def main():

    if len(sys.argv) < 1:
        print("Must provide simulation name")
        sys.exit(1)
    sim = sys.argv[1]

    # z=0 snapshot
    if "TNG" in sim:
        snap_num = 99
    elif "Illustris" in sim:
        snap_num = 135
    else:
        print("Unknown simulation")
        sys.exit()

    subhalo_fields = ["SubhaloPos", "SubhaloMassType", "SubhaloSFR"] \
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
    halo_data = tng.groupcat.loadHalos(sim, snap_num, fields=group_fields)

    sh_dm_mass = subhalo_data["SubhaloMassType"][:,1] * 1e10
    sh_st_mass = subhalo_data["SubhaloMassType"][:,4] * 1e10
    sh_tot_mass = np.sum(subhalo_data["SubhaloMassType"], axis=1) * 1e10
    if "Illustris" not in sim:
        flag = subhalo_data["SubhaloFlag"].astype(bool)
        print(np.sum(~flag[(sh_st_mass > gal_mass_cut) * (sh_dm_mass / sh_tot_mass > 0.1)]))
    else:
        flag = sh_dm_mass / sh_tot_mass > 0.1

    galaxy = (sh_st_mass > gal_mass_cut) * flag
    ssfr = subhalo_data["SubhaloSFR"][galaxy] / sh_st_mass[galaxy]

    if not match_ssfr:
        sf = ssfr > ssfr_cut
        qsnt = ssfr <= ssfr_cut
    else:
        qsnt = np.argsort(ssfr)[:len(ssfr)//2]
        sf = np.argsort(ssfr)[len(ssfr)//2:]
        print(f"sSFR cut: {np.max(ssfr[qsnt]):.3e}")

    pos_gal = subhalo_data["SubhaloPos"][galaxy] / 1e3
    pos_sfgal = pos_gal[sf]
    pos_qsntgal = pos_gal[qsnt]

    # use abundance matching to get halos
    m_c200 = halo_data["Group_M_Crit200"] * 1e10
    halo = np.isin(np.arange(len(m_c200)), np.argsort(m_c200)[::-1][:len(pos_gal)])
    pos_halo = halo_data["GroupPos"][halo] / 1e3

    n_selections = []
    lbars = []
    es_all = []
    es_scaled_all = []
    xi_all = []
    for i, pos in enumerate([pos_gal, pos_halo, pos_sfgal, pos_qsntgal]):
        print("Starting", selections[i])
        n_selections.append(len(pos))
        print(len(pos), "objects selected")
        lbar = boxsize / np.cbrt(len(pos))
        lbars.append(lbar)

        print("Computing 2pcf..")
        rbins = np.logspace(np.log10(lbar/2), np.log10(boxsize/3.1), 35)
        sample_xi = xi(boxsize, 2, rbins, *pos.T, output_ravg=True)
        xi_all.append(sample_xi)

        print("Computing topology...")
        pairs = calc_persistence(pos, boxsize=boxsize, precision="fast")
        pairs = [np.array(pairs[d]) for d in range(3)]

        print("Calculating ES curves")
        es = ES.fit_transform(DS.fit_transform(pairs))
        es_all.append(es)

        print("Scaling and interpolating ES curves")
        interp = interp1d(alpha/lbar, es, axis=-1, bounds_error=False, fill_value=0, assume_sorted=True, kind="slinear")
        es_scaled = interp(alpha_scaled)
        norm = np.trapz(np.abs(es_scaled), alpha_scaled, axis=-1)
        es_scaled /= np.expand_dims(norm, -1)
        es_scaled_all.append(es_scaled)

    sim_name = sim.split('/')[-1]
    fname = f"topology_summaries/{sim_name}/es_all.npz"
    print(f"Saving data to {fname}")
    np.savez(fname, alpha=alpha, alpha_scaled=alpha_scaled, n_selections=n_selections, lbars=lbars,
            xi_gal=xi_all[0], xi_halo=xi_all[1], xi_sfgal=xi_all[2], xi_qsntgal=xi_all[3],
            es_gal=es_all[0], es_scaled_gal=es_scaled_all[0], es_halo=es_all[1], es_scaled_halo=es_scaled_all[1],
            es_sfgal=es_all[2], es_scaled_sfgal=es_scaled_all[2], es_qsntgal=es_all[3],
            es_scaled_qsntgal=es_scaled_all[3])


main()
