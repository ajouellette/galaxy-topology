import glob
import os
import sys

import gudhi.representations as gdr
import h5py
import illustris_python as tng
import numpy as np
from alpha_complex_periodic import calc_persistence
from mpi4py import MPI
from scipy.interpolate import interp1d


def load_camels(sim_name, snap_num=-1):
    """Load a CAMELS-IllustrisTNG sim."""

    if snap_num == -1:
        files = glob.glob(sim_name + "/fof_subhalo_tab_*.hdf5")
        if len(files) == 0:
            raise ValueError("No snapshot files found in "+sim_name)
        files.sort()
        fname = files[-1]
    else:
        fname = sim_name + f"/fof_subhalo_tab_{snap_num:03}.hdf5"
    data = {}
    with h5py.File(fname) as f:
        data["boxsize"] = f["Header"].attrs["BoxSize"] / 1e3
        data["HaloPos"] = f["Group/GroupPos"][:] / 1e3
        data["HaloMass"] = f["Group/Group_M_Crit200"][:] * 1e10
        data["SubhaloPos"] = f["Subhalo/SubhaloPos"][:] / 1e3
        data["SubhaloMass"] = f["Subhalo/SubhaloMassType"][:] * 1e10
        data["SubhaloSFR"] = f["Subhalo/SubhaloSFR"][:]

    return data


def load_illustris(sim_name, snap_num=-1):
    """Load an Illustris or TNG sim."""

    if snap_num == -1:
        snaps = glob.glob(sim_name + "/groups_*")
        if len(snaps) == 0:
            raise ValueError("No group directories found in "+sim_name)
        snaps.sort()
        snap_num = int(snaps[-1].split('_')[-1])
    header = tng.groupcat.loadHeader(sim_name, snap_num)
    halo_data = tng.groupcat.loadHalos(sim_name, snap_num,
            ["GroupPos", "Group_M_Crit200"])
    subhalo_data = tng.groupcat.loadSubhalos(sim_name, snap_num,
            ["SubhaloPos", "SubhaloMassType", "SubhaloSFR"])
    data = {"boxsize": header["BoxSize"] / 1e3,
            "HaloPos": halo_data["GroupPos"] / 1e3, "HaloMass": halo_data["Group_M_Crit200"] * 1e10,
            "SubhaloPos": subhalo_data["SubhaloPos"] / 1e3, "SubhaloMass": subhalo_data["SubhaloMassType"] * 1e10,
            "SubhaloSFR": subhalo_data["SubhaloSFR"]}

    return data


def load_sam(sim_name, snap_num=-1):
    """Load a CAMELS-SAM sim."""

    base_name = "/../rockstar_" if "1P" in sim_name else "/rockstar_"
    if snap_num == -1:
        rockstar = glob.glob(sim_name + base_name + "*.hdf5")
        if len(rockstar) == 0:
            raise ValueError("No rockstar files found in "+sim_name)
        rockstar.sort()
        rockstar_fname = rockstar[-1]
        sam = glob.glob(sim_name + "/sc_sam_tab_*.hdf5")
        if len(sam) == 0:
            raise ValueError("No sam files found in "+sim_name)
        sam.sort()
        sam_fname = sam[-1]
    else:
        rockstar_fname = sim_name + base_name + f"{snap_num:02}.hdf5"
        sam_fname = sim_name + f"/sc_sam_tab_{snap_num:02}.hdf5"

    h = 0.6711
    data = {}
    with h5py.File(rockstar_fname) as f:
        data["boxsize"] = f.attrs["boxsize"]
        data["HaloPos"] = f["pos"][:]
        data["HaloMass"] = f["M200c"][:]
    with h5py.File(sam_fname) as f:
        data["SubhaloPos"] = f["galprop/pos"][:] * h
        data["SubhaloPos"][data["SubhaloPos"] < 0] += data["boxsize"]
        data["SubhaloPos"][data["SubhaloPos"] >= data["boxsize"]] -= data["boxsize"]
        data["SubhaloMass"] = np.zeros((len(data["SubhaloPos"]), 5))
        data["SubhaloMass"][:,4] = f["galprop/mstar"][:] * 1e9 * h
        data["SubhaloHaloMass"] = f["galprop/mhalo"][:] * 1e9 * h
        data["SubhaloHaloIndex"] = f["galprop/halo_index"][:]
        data["SubhaloSFR"] = f["galprop/sfr"][:]
        data["SubhaloCentral"] = f["galprop/sat_type"][:] == 0

    return data


DS = gdr.preprocessing.DiagramSelector(use=True)


def calc_summary(point_sets, summary, boxsize=None):
    pairs = []
    for points in point_sets:
        try:
            pairs.append(calc_persistence(points, boxsize=boxsize, precision="fast"))
        except ValueError:
            pairs.append(3 * [np.zeros((1,2))])  # fake data, should result in NaNs in summary
    pairs = [[np.array(p[d]) for d in range(3)] for p in pairs]
    return np.array([summary.fit_transform(DS.fit_transform(p)) for p in pairs])


def camels_sam_params(sam_params):
    """Get CAMELS parameter values from SAM parameter values."""

    params = np.copy(sam_params[:,:5])
    params[:,2] /= 1.7
    params[:,3] -= 3
    params[:,4] /= 0.002
    return params


def main():
    alpha_resolution_factor = 50  # number of alpha grid points per Mpc
    scaled_range = [0, 5]  # range of alpha / l
    scaled_resolution = 500  # resolution of alpha / l
    alpha_scaled = np.linspace(*scaled_range, scaled_resolution)

    min_halo_mass_cut = 5e10  # minimum mass cut when selecting DM halos
    halo_match = True  # match number of galaxies and halos
    halo_random_sample = True  # randomly sample halos or pick N most massive

    st_mass_cut = 2e8  # min stellar mass cut for galaxies
    dm_frac_cut = 0.1  # min DM mass fraction for galaxies
    max_halo_cut = None  # max halo mass for galaxies
    satellites = None  # whether to consider only satellites (centrals)

    ssfr_cut = 10**-11  # sSFR cut for quiescent/star-forming
    ssfr_match = False  # whether to match number of quiescent/star-forming galaxies

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    suite = sys.argv[1]
    if "Illustris" in suite or "TNG" in suite:
        sim_set = None
        sims = [suite,]
    if "camels" in suite:
        sim_set = sys.argv[2]
        if sim_set not in ["CV", "1P", "LH", "EX"]:
            raise ValueError("Unknown simulation set.")

        if sim_set == "EX":
            sims = [f"{suite}/{sim_set}_0", f"{suite}/{sim_set}_3"]
        else:
            sims = glob.glob(suite + '/' + sim_set + "_*")
            # remove duplicate sims in 1P group
            sim_dups = ["1P_16", "1P_27", "1P_38", "1P_49", "1P_60"]
            for dup in sim_dups:
                dup_fname = suite + '/' + dup
                if dup_fname in sims:
                    sims.remove(dup_fname)

        if "sam" in suite and sim_set == "1P":
            sims = glob.glob(suite + "/CV_*/1P_*")

    if len(sims) == 0:
        raise ValueError("No simulations found matching given criteria")

    if rank == 0:
        print(f"{rank} Found {len(sims)} sims")

    avg, res = divmod(len(sims), n_ranks)
    count = [avg + 1 if r < res else avg for r in range(n_ranks)]
    offset = [sum(count[:r]) for r in range(n_ranks)]

    params_all = []
    n_selected_all = []
    lbars_all = []
    es_all = []
    es_scaled_all = []
    halo_mass_cuts = []
    ssfr_cuts = []

    start_i = offset[rank]
    end_i = start_i + count[rank]
    for sim in sims[start_i:end_i]:
        print("Starting", sim)

        try:
            params_all.append(np.loadtxt(sim + "/CosmoAstro_params.txt"))
        except FileNotFoundError:
            print("Could not find", sim+"/CosmoAstro_params.txt")
            params_all.append([0, 0, 0, 0, 0, 0])

        if "camels-sam" in sim:
            data = load_sam(sim)
        elif "camels" in sim:
            data = load_camels(sim)
        elif "Illustris" in sim or "TNG" in sim:
            data = load_illustris(sim)

        boxsize = data["boxsize"]
        alpha_range = [0, boxsize/2]
        alpha_resolution = int(boxsize/2 * alpha_resolution_factor)
        alpha = np.linspace(*alpha_range, alpha_resolution)

        ES = gdr.Entropy(mode="vector", resolution=alpha_resolution, sample_range=alpha_range,
                normalized=False)

        if "sam" not in sim:
            tot_sh_mass = np.sum(data["SubhaloMass"], axis=1)
            # stellar mass cut and dm mass fraction cut
            galaxy = (data["SubhaloMass"][:,4] > st_mass_cut) \
                    * (data["SubhaloMass"][:,1] / tot_sh_mass > dm_frac_cut)
        else:
            # for SAM, only need stellar mass cut
            galaxy = data["SubhaloMass"][:,4] > st_mass_cut
            # filter out galaxy clusters
            if max_halo_cut is not None:
                halo_indicies = data["SubhaloHaloMass"] >= max_halo_cut
                galaxy *= ~np.isin(data["SubhaloHaloIndex"], halo_indicies)
            # filter out satellites
            if satellites is not None:
                if satellites:
                    galaxy *= ~data["SubhaloCentral"]
                else:
                    galaxy *= data["SubhaloCentral"]
        galaxy_selection = data["SubhaloPos"][galaxy]

        if not halo_match:
            halo_selection = data["HaloPos"][data["HaloMass"] > min_halo_mass_cut]
        else:
            n_halos = len(galaxy_selection)
            if halo_random_sample:
                halo = np.nonzero(data["HaloMass"] > min_halo_mass_cut)[0]
                halo = np.random.choice(halo, size=min(len(halo), n_halos), replace=False)
            else:
                halo = np.isin(np.arange(len(data["HaloPos"])),
                        np.argsort(data["HaloMass"])[::-1][:n_halos])

            halo_selection = data["HaloPos"][halo]
            min_halo_mass_cut = np.min(data["HaloMass"][halo])
            halo_mass_cuts.append(min_halo_mass_cut)
            print(f"Halo mass cut: {min_halo_mass_cut:.3e}")

        ssfr = data["SubhaloSFR"][galaxy] / data["SubhaloMass"][galaxy][:,4]
        if not ssfr_match:
            sf_selection = galaxy_selection[ssfr > ssfr_cut]
            qsnt_selection = galaxy_selection[ssfr <= ssfr_cut]
        else:
            ssfr_sort = np.argsort(ssfr)
            sf_selection = galaxy_selection[ssfr_sort[len(ssfr)//2:]]
            qsnt_selection = galaxy_selection[ssfr_sort[:len(ssfr)//2]]
            ssfr_cut = ssfr[len(ssfr)//2]
            ssfr_cuts.append(ssfr_cut)
            print(f"sSFR cut: {ssfr_cut:.3e}")

        pos_list = [halo_selection, galaxy_selection, sf_selection, qsnt_selection]
        n_selected = np.array([len(pos) for pos in pos_list])
        n_selected_all.append(n_selected)
        print("Selections:", n_selected)

        print("Computing topological summaries...")
        es = calc_summary(pos_list, ES, boxsize=boxsize)
        norm = np.trapz(np.abs(es), alpha, axis=-1)
        es /= np.expand_dims(norm, -1)
        es_all.append(es)

        print("Scaling and interpolating summaries...")
        lbars = np.array([boxsize/np.cbrt(n) for n in n_selected])
        lbars_all.append(lbars)
        es_scaled = np.zeros((len(es), 3, scaled_resolution))
        for i in range(len(es)):
            interp = interp1d(alpha/lbars[i], es[i], axis=-1, bounds_error=False, fill_value=0,
                    assume_sorted=True)
            es_scaled[i] = interp(alpha_scaled)
        norm = np.trapz(np.abs(es_scaled), alpha_scaled, axis=-1)
        es_scaled /= np.expand_dims(norm, -1)
        es_scaled_all.append(es_scaled)

    params_all = np.array(params_all)
    n_selected_all = np.array(n_selected_all)
    lbars_all = np.array(lbars_all)
    es_all = np.array(es_all)
    es_scaled_all = np.array(es_scaled_all)
    halo_mass_cuts = np.array(halo_mass_cuts)
    ssfr_cuts = np.array(ssfr_cuts)

    comm.Barrier()
    if rank == 0:
        print("Collecting data...")

    params_all = comm.gather(params_all)
    n_selected_all = comm.gather(n_selected_all)
    lbars_all = comm.gather(lbars_all)
    es_all = comm.gather(es_all)
    es_scaled_all = comm.gather(es_scaled_all)
    if halo_match:
        halo_mass_cuts = comm.gather(halo_mass_cuts)
    if ssfr_match:
        ssfr_cuts = comm.gather(ssfr_cuts)

    if rank == 0:
        params = np.vstack(params_all)
        if "camels-sam" in suite:
            params = camels_sam_params(params)
        n_selected = np.vstack(n_selected_all)
        lbars = np.vstack(lbars_all)
        es = np.vstack(es_all)
        es_scaled = np.vstack(es_scaled_all)
        if halo_match:
            halo_mass_cuts = np.hstack(halo_mass_cuts)
        if ssfr_match:
            ssfr_cuts = np.hstack(ssfr_cuts)

        suite_name = suite.split('/')[-1]
        save_dir = f"topology_summaries/{suite_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_fname = save_dir + f"/es{'_'+sim_set if sim_set is not None else ''}_all.npz"
        print(f"Saving data to {save_fname}")
        np.savez(save_fname, params=params, alpha=alpha, alpha_scaled=alpha_scaled,
                es=es, es_scaled=es_scaled, n_selected=n_selected, lbars=lbars,
                halo_mass_cut=halo_mass_cuts if halo_match else min_halo_mass_cut,
                gal_ssfr_cut=ssfr_cuts if ssfr_match else ssfr_cut,
                gal_st_mass_cut=st_mass_cut, gal_dm_frac_cut=dm_frac_cut)


if __name__ == "__main__":
    main()
