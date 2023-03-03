import glob
import os
import sys

from Corrfunc.theory.xi import xi as calc_xi
import gudhi.representations as gdr
import numpy as np
import h5py
from alpha_complex_periodic import calc_persistence
from mpi4py import MPI
from scipy.interpolate import interp1d
from utils import load_camels, load_illustris, load_sam


DS = gdr.preprocessing.DiagramSelector(use=True)


def calc_summary(point_sets, summary, boxsize=None):
    pairs = []
    for points in point_sets:
        try:
            pairs.append(calc_persistence(points, boxsize=boxsize, precision="fast"))
        except ValueError:
            # fake data, should result in NaNs in summary
            pairs.append(3 * [np.zeros((1,2))])
    pairs = [[np.array(p[d]) for d in range(3)] for p in pairs]
    if isinstance(summary, list):
        return [np.array([func.fit_transform(DS.fit_transform(p)) for p in pairs]) for func in summary]

    return np.array([summary.fit_transform(DS.fit_transform(p)) for p in pairs])


def scale_summary(alpha, summary, lbar, alpha_scaled):
    interp = interp1d(alpha/lbar, summary, axis=-1, bounds_error=False, fill_value=0, assume_sorted=True)
    return interp(alpha_scaled) * lbar**3


def camels_sam_params(sam_params):
    """Get CAMELS parameter values from SAM parameter values."""

    params = np.copy(sam_params[:,:5])
    params[:,2] /= 1.7
    params[:,3] -= 3
    params[:,4] /= 0.002
    return params


def main():
    max_alpha = 0.2  # maximum alpha value as fraction of box size
    alpha_resolution_factor = 50  # number of alpha grid points per Mpc
    scaled_range = [2e-2, 4]  # range of alpha / l
    scaled_resolution = 250  # resolution of alpha / l
    log_scaled = True  # evaluate scaled summary on a log grid
    if log_scaled:
        alpha_scaled = np.logspace(*np.log10(scaled_range), scaled_resolution)
    else:
        alpha_scaled = np.linspace(*scaled_range, scaled_resolution)

    min_halo_mass_cut = 10**10.5  # minimum mass cut when selecting DM halos
    halo_mass_bins = [10.5, 11, 11.5, 12, 12.5]  # halo mass bins in terms of log Mvir

    st_mass_cut = 5e8  # min stellar mass cut for galaxies
    dm_frac_cut = 0.1  # min DM mass fraction for galaxies

    ssfr_cut = 10**-10.5  # sSFR cut for quiescent/star-forming

    calc_2pcf = True  # whether to calculate 2-point correlation functions
    n_rbins = 30  # number of r bins for 2pcf (per log Mpc)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    suite = sys.argv[1]
    if "Illustris" in suite or "TNG" in suite:
        if len(sys.argv) > 2:
            save_suffix = sys.argv[2]
        else:
            save_suffix = None
        sim_set = None
        sims = [suite,]
    if "camels" in suite:
        sim_set = sys.argv[2]
        if len(sys.argv) > 3:
            save_suffix = sys.argv[3]
        else:
            save_suffix = None
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
    xi_all = []
    xi_ravg_all = []
    lbars_all = []
    bc_all = []
    bc_scaled_all = []
    halo_mass_cuts = []

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
        alpha_range = [0, boxsize * max_alpha]
        alpha_resolution = int(boxsize * max_alpha * alpha_resolution_factor)
        alpha = np.linspace(*alpha_range, alpha_resolution)

        BC = gdr.BettiCurve(predefined_grid=alpha)

        if "sam" not in sim:
            tot_sh_mass = np.sum(data["SubhaloMass"], axis=1)
            # stellar mass cut and dm mass fraction cut
            galaxy = (data["SubhaloMass"][:,4] > st_mass_cut) \
                    * (data["SubhaloMass"][:,1] / tot_sh_mass > dm_frac_cut)
        else:
            # for SAM, only need stellar mass cut
            galaxy = data["SubhaloMass"][:,4] > st_mass_cut

        galaxy_selection = data["SubhaloPos"][galaxy]

        halo_selection = data["HaloPos"][data["HaloMass"] > min_halo_mass_cut

        ssfr = data["SubhaloSFR"][galaxy] / data["SubhaloMass"][galaxy][:,4]
        sf_selection = galaxy_selection[ssfr > ssfr_cut]
        qsnt_selection = galaxy_selection[ssfr <= ssfr_cut]

        pos_list = [halo_selection, galaxy_selection, sf_selection, qsnt_selection]
        n_selected = np.array([len(pos) for pos in pos_list])
        n_selected_all.append(n_selected)
        print("Selections:", n_selected)

        if calc_2pcf:
            print("Computing 2PCFs...")
            rbins = np.logspace(-0.1, np.log10(boxsize/5), int(n_rbins*(np.log10(boxsize/5)+0.1)))
            xi = np.zeros((len(pos_list), n_rbins-1))
            xi_ravg = np.zeros_like(xi)
            for i, points in enumerate(pos_list):
                xi_sample = calc_xi(boxsize, 1, rbins, *points.T, output_ravg=True)
                xi[i] = xi_sample["xi"]
                xi_ravg[i] = xi_sample["ravg"]
            xi_all.append(xi)
            xi_ravg_all.append(xi_ravg)

        print("Computing topological summaries...")
        bc = calc_summary(pos_list, BC, boxsize=boxsize) / boxsize**3
        bc_all.append(bc)

        print("Scaling and interpolating summaries...")
        lbars = np.array([boxsize/np.cbrt(n) for n in n_selected])
        lbars_all.append(lbars)
        bc_scaled = np.zeros((len(bc), 3, scaled_resolution))
        for i in range(len(bc)):
            bc_scaled[i] = scale_summary(alpha, bc, lbars[i], alpha_scaled)
        bc_scaled_all.append(bc_scaled)

    params_all = np.array(params_all)
    n_selected_all = np.array(n_selected_all)
    xi_all = np.array(xi_all)
    xi_ravg_all = np.array(xi_ravg_all)
    lbars_all = np.array(lbars_all)
    bc_all = np.array(bc_all)
    bc_scaled_all = np.array(bc_scaled_all)

    comm.Barrier()
    if rank == 0:
        print("Collecting data...")

    params_all = comm.gather(params_all)
    n_selected_all = comm.gather(n_selected_all)
    if calc_2pcf:
        xi_all = comm.gather(xi_all)
        xi_ravg_all = comm.gather(xi_ravg_all)
    lbars_all = comm.gather(lbars_all)
    bc_all = comm.gather(bc_all)
    bc_scaled_all = comm.gather(bc_scaled_all)

    if rank == 0:
        params = np.vstack(params_all)
        if "camels-sam" in suite:
            params = camels_sam_params(params)
        n_selected = np.vstack(n_selected_all)
        if calc_2pcf:
            xi_all = np.vstack(xi_all)
            xi_ravg_all = np.vstack(xi_ravg_all)
        lbars = np.vstack(lbars_all)
        bc = np.vstack(bc_all)
        bc_scaled = np.vstack(bc_scaled_all)

        suite_name = suite.split('/')[-1]
        save_dir = f"topology_summaries/{suite_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_fname = save_dir + f"/es{'_'+sim_set if sim_set is not None else ''}_all{'_'+save_suffix if save_suffix is not None else ''}.npz"
        print(f"Saving data to {save_fname}")
        np.savez(save_fname, params=params, alpha=alpha, alpha_scaled=alpha_scaled,
                bc=bc, bc_scaled=bc_scaled, n_selected=n_selected, lbars=lbars,
                xi_ravg=xi_ravg_all if calc_2pcf else None,
                xi=xi_all if calc_2pcf else None)


if __name__ == "__main__":
    main()
