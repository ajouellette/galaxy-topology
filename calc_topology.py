import glob
import sys

from Corrfunc.theory.xi import xi as calc_xi
import gudhi.representations as gdr
import numpy as np
import h5py
from alpha_complex_periodic import calc_persistence
from mpi4py import MPI
from scipy.interpolate import interp1d


def select_halos(data, min_mass, bins=None, include_upper_bin=True):
    inds = np.nonzero(data["HaloMass"] > min_mass)[0]

    if bins is None:
        return [data["HaloPos"][inds],]

    halos_binned = []
    log_mass = np.log10(data["HaloMass"][inds])
    pos = data["HaloPos"][inds]
    bin_inds = np.digitize(log_mass, bins)
    for bin_i in range(len(bins) if not include_upper_bin else len(bins)+1):
        bin_mask = bin_inds == bin_i
        halos_binned.append(pos[bin_mask])

    return halos_binned


def select_galaxies(data, min_mass, min_dm_frac=0.1, ssfr_cut=10**-10.5):
    galaxies = data["SubhaloStMass"] > min_mass
    if "SubhaloDmFrac" in data.keys():
        galaxies *= data["SubhaloDmFrac"] > min_dm_frac
    galaxies = np.nonzero(galaxies)[0]
    all_pos = data["SubhaloPos"][galaxies]
    all_type = data["SubhaloCentral"][galaxies]

    ssfr = data["SubhaloSFR"][galaxies] / data["SubhaloStMass"][galaxies]
    sf = ssfr > ssfr_cut

    return [all_pos[sf*all_type], all_pos[sf*(~all_type)],
            all_pos[(~sf)*all_type], all_pos[(~sf)*(~all_type)]]


def calc_summary(point_sets, summary, boxsize=None, exclude_inf=True):
    pairs = []
    for points in point_sets:
        try:
            pairs.append(calc_persistence(points, boxsize=boxsize, precision="fast"))
        except ValueError:
            # fake data, should result in NaNs in summary
            pairs.append(3 * [np.zeros((1,2))])
    pairs = [[np.array(p[d]) for d in range(3)] for p in pairs]
    DS = gdr.preprocessing.DiagramSelector(use=exclude_inf)
    if isinstance(summary, list):
        return [np.array([func.fit_transform(DS.fit_transform(p)) for p in pairs]) for func in summary]

    return np.array([summary.fit_transform(DS.fit_transform(p)) for p in pairs])


def scale_summary(alpha, summary, lbar, alpha_scaled):
    interp = interp1d(alpha/lbar, summary, axis=-1,
            bounds_error=False, fill_value=0, assume_sorted=True)
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
    halo_mass_bins = [11, 11.5, 12, 12.5]  # halo mass bins in terms of log Mvir

    st_mass_cut = 5e8  # min stellar mass cut for galaxies
    dm_frac_cut = 0.1  # min DM mass fraction for galaxies

    ssfr_cut = 10**-10.5  # sSFR cut for quiescent/star-forming

    calc_2pcf = True  # whether to calculate 2-point correlation functions
    n_rbins = 30  # number of r bins for 2pcf (per log Mpc)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    sim = sys.argv[1]
    selection = sys.argv[2]
    selections = ["halos", "galaxies"]
    if selection not in selections:
        raise ValueError(f"selection must be one of {selections}")

    snaps = glob.glob(sim + "/*_reduced.hdf5")
    if len(snaps) == 0:
        if rank == 0:
            print("No snapshots found in {sim}")
        sys.exit()

    if rank == 0:
        print(f"Found {len(snaps)} snapshots")

    avg, res = divmod(len(snaps), n_ranks)
    count = [avg + 1 if r < res else avg for r in range(n_ranks)]
    offset = [sum(count[:r]) for r in range(n_ranks)]

    redshifts = []
    n_selected_all = []
    xi_all = []
    xi_ravg_all = []
    lbars_all = []
    bc_all = []
    bc_scaled_all = []

    start_i = offset[rank]
    end_i = start_i + count[rank]
    for snap in snaps[start_i:end_i]:
        print("Starting snapshot", snap)

        snap_data = {}
        with h5py.File(snap) as f:
            boxsize = f.attrs["boxsize"]
            redshift = f.attrs["redshift"]
            redshifts.append(redshift)
            for field in f.keys():
                snap_data[field] = f[field][:]

        alpha_range = [0, boxsize * max_alpha]
        alpha_resolution = int(boxsize * max_alpha * alpha_resolution_factor)
        alpha = np.linspace(*alpha_range, alpha_resolution)

        BC = gdr.BettiCurve(predefined_grid=alpha)

        if selection == "halos":
            samples = select_halos(snap_data, min_halo_mass_cut, halo_mass_bins)
        elif selection == "galaxies":
            samples = select_galaxies(snap_data, st_mass_cut, dm_frac_cut, ssfr_cut)

        n_selected = np.array([len(sample) for sample in samples])
        n_selected_all.append(n_selected)
        print("Selections:", n_selected)

        if calc_2pcf:
            print("Computing 2PCFs...")
            rbins = np.logspace(-0.1, np.log10(boxsize/5), int(n_rbins*(np.log10(boxsize/5)+0.1)))
            xi = np.zeros((len(samples), len(rbins)-1))
            xi_ravg = np.zeros_like(xi)
            for i, sample in enumerate(samples):
                xi_sample = calc_xi(boxsize, 1, rbins, *sample.T, output_ravg=True)
                xi[i] = xi_sample["xi"]
                xi_ravg[i] = xi_sample["ravg"]
            xi_all.append(xi)
            xi_ravg_all.append(xi_ravg)

        print("Computing Betti curves...")
        bc = calc_summary(samples, BC, boxsize=boxsize) / boxsize**3
        bc_all.append(bc)

        print("Scaling and interpolating Betti curves...")
        lbars = boxsize / np.cbrt(n_selected)
        lbars_all.append(lbars)
        bc_scaled = np.zeros((len(bc), 3, scaled_resolution))
        for i in range(len(bc)):
            bc_scaled[i] = scale_summary(alpha, bc[i], lbars[i], alpha_scaled)
        bc_scaled_all.append(bc_scaled)

    redshifts = np.array(redshifts)
    n_selected_all = np.array(n_selected_all)
    xi_all = np.array(xi_all)
    xi_ravg_all = np.array(xi_ravg_all)
    lbars_all = np.array(lbars_all)
    bc_all = np.array(bc_all)
    bc_scaled_all = np.array(bc_scaled_all)

    comm.Barrier()
    if rank == 0:
        print("Collecting data...")

    redshifts = comm.gather(redshifts)
    n_selected_all = comm.gather(n_selected_all)
    if calc_2pcf:
        xi_all = comm.gather(xi_all)
        xi_ravg_all = comm.gather(xi_ravg_all)
    lbars_all = comm.gather(lbars_all)
    bc_all = comm.gather(bc_all)
    bc_scaled_all = comm.gather(bc_scaled_all)

    if rank == 0:
        redshifts = np.hstack(redshifts)
        n_selected = np.vstack(n_selected_all)
        if calc_2pcf:
            xi_all = np.vstack(xi_all)
            xi_ravg_all = np.vstack(xi_ravg_all)
        lbars = np.vstack(lbars_all)
        bc = np.vstack(bc_all)
        bc_scaled = np.vstack(bc_scaled_all)

        save_fname = sim + f"/betti_{selection}.npz"
        print(f"Saving data to {save_fname}")
        np.savez(save_fname, redshifts=redshifts, alpha=alpha, alpha_scaled=alpha_scaled,
                n_selected=n_selected, lbars=lbars,
                bc=bc, bc_scaled=bc_scaled,
                xi_ravg=xi_ravg_all if calc_2pcf else None, xi=xi_all if calc_2pcf else None)


if __name__ == "__main__":
    main()
