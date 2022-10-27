import sys
import os
import glob
import numpy as np
from scipy.interpolate import interp1d
import h5py
from mpi4py import MPI
import gudhi as gd
import gudhi.representations as gdr
from alpha_complex_periodic import calc_persistence


alpha_resolution = 1000
alpha_range = [0, 10]
alpha = np.linspace(*alpha_range, alpha_resolution)
scaled_resolution = 500
scaled_range = [0, 4]
alpha_scaled = np.linspace(*scaled_range, scaled_resolution)

# Topological summaries
DS = gdr.preprocessing.DiagramSelector(use=True)
ES = gdr.Entropy(mode="vector", resolution=alpha_resolution, sample_range=alpha_range,
        normalized=False)


def calc_summary(point_sets, summary, boxsize=None):
    pairs = []
    for points in point_sets:
        try:
            pairs.append(calc_persistence(points, boxsize=boxsize, precision="fast"))
        except ValueError:
            pairs.append(3 * [np.zeros((1,2))])
            #pairs.append(3 * [np.array([[0, np.inf]])])
    pairs = [[np.array(p[d]) for d in range(3)] for p in pairs]
    return np.array([summary.fit_transform(DS.fit_transform(p)) for p in pairs])


# all CAMELS sims have boxsize of 25 Mpc
boxsize = 25

# selection cuts
n_star_cut = 10
mass_cut = 2e8
#ssfr_cut = 10**-10.5
ssfr_cut = 10**-10

# possible selections
selections = ["galaxy", "halo", "sfgal", "qsntgal"]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if len(sys.argv) < 3:
        if rank == 0:
            print("Must provide simulation suite name (optionally simulation set name) and object selection")
        sys.exit(1)
    suite = sys.argv[1]
    if len(sys.argv) > 3:
        sim_set = sys.argv[2]
        selection = sys.argv[3]
    else:
        selection = sys.argv[2]
        sim_set = None

    if selection not in selections:
        if rank == 0:
            print("selection must be one of", selections)
        sys.exit(1)

    if sim_set:
        sims = glob.glob(suite + '/' + sim_set + "_*")
    else:
        sims = glob.glob(suite + "/*_*")

    # remove duplicate sims in 1P group
    sim_dups = ["1P_16", "1P_27", "1P_38", "1P_49", "1P_60"]
    for dup in sim_dups:
        dup_fname = suite + '/' + dup
        if dup_fname in sims:
            sims.remove(dup_fname)

    if rank == 0:
        print(f"Found {len(sims)} sims")

    avg, rem = divmod(len(sims), n_ranks)
    count = [avg+1 if r < rem else avg for r in range(n_ranks)]
    offset = [sum(count[:r]) for r in range(n_ranks)]

    print(f"{rank} Loading sims")

    pos_list = []
    params = []
    for sim_i in range(offset[rank], offset[rank]+count[rank]):
        sim = sims[sim_i]

        params.append(np.loadtxt(sim + "/CosmoAstro_params.txt"))

        with h5py.File(sim + "/fof_subhalo_tab_033.hdf5") as f:
            pos = f["Subhalo/SubhaloPos"][:] / 1e3
            n_stars = f["Subhalo/SubhaloLenType"][:,4]
            mass = f["Subhalo/SubhaloMass"][:] * 1e10
            st_mass = f["Subhalo/SubhaloMassType"][:,4] * 1e10
            dm_mass = f["Subhalo/SubhaloMassType"][:,1] * 1e10

            galaxy = (n_stars > n_star_cut) * (st_mass > mass_cut)
            # use abundance matching to get halos
            halo = np.isin(np.arange(len(pos)), np.argsort(dm_mass)[::-1][:np.sum(galaxy)])

            ssfr = f["Subhalo/SubhaloSFR"][galaxy] / st_mass[galaxy]

            sf = ssfr > ssfr_cut
            qsnt = ssfr <= ssfr_cut

            if selection == "galaxy":
                pos = pos[galaxy]
            elif selection == "halo":
                pos = pos[halo]
            elif selection == "sfgal":
                pos = pos[galaxy][sf]
            elif selection == "qsntgal":
                pos = pos[galaxy][qsnt]

            pos_list.append(pos)

    params = np.vstack(params)

    n_selection = np.array([len(p) for p in pos_list])
    print(n_selection)

    print(f"{rank} Calculating ES curves")
    es = calc_summary(pos_list, ES, boxsize=boxsize)

    print(f"{rank} Scaling and interpolating ES curves")
    es_scaled = np.zeros((len(es), 3, scaled_resolution))
    for i in range(len(es)):
        lbar = boxsize/np.cbrt(n_selection[i])
        interp = interp1d(alpha/lbar, es[i], axis=-1, bounds_error=False, fill_value=0, assume_sorted=True)
        es_scaled[i] = interp(alpha_scaled)

    norm = np.trapz(np.abs(es_scaled), alpha_scaled, axis=-1)
    es_scaled /= np.expand_dims(norm, -1)

    comm.Barrier()
    if rank == 0:
        print("Collecting data")

    params = comm.gather(params)
    n_selection = comm.gather(n_selection)
    es = comm.gather(es)
    es_scaled = comm.gather(es_scaled)

    if rank == 0:
        suite_name = suite.split('/')[-1]
        fname = f"topology_summaries/{suite_name}/es{'_'+sim_set if sim_set else ''}_{selection}2.npz"
        print(f"Saving data to {fname}")

        params = np.vstack(params)
        n_selection = np.hstack(n_selection)
        es = np.vstack(es)
        es_scaled = np.vstack(es_scaled)
        lbar = boxsize / np.cbrt(n_selection)

        np.savez(fname, alpha=alpha, es=es, alpha_scaled=alpha_scaled, es_scaled=es_scaled, params=params, n_selection=n_selection, lbar=lbar)


main()
