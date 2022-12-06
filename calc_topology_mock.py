import sys
import os
import glob
import numpy as np
from scipy.interpolate import interp1d
from mpi4py import MPI
import gudhi.representations as gdr
from alpha_complex_periodic import calc_persistence
from calc_topology import calc_summary


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print("Must provide file for mock data")
        sys.exit(1)
    fname = sys.argv[1]
    data = np.load(fname, allow_pickle=True)

    boxsize = data["boxsize"]
    pos_list = data["data"]
    lbar_vals = data["lbar"]

    alpha_range = [0, boxsize/2]
    alpha_resolution = int(alpha_range[-1] * 50)
    alpha = np.linspace(*alpha_range, alpha_resolution)
    scaled_resolution = 500
    scaled_range = [0, 5]
    alpha_scaled = np.linspace(*scaled_range, scaled_resolution)

    # Topological summaries
    DS = gdr.preprocessing.DiagramSelector(use=True)
    SIL1 = gdr.Silhouette(weight=lambda x: np.power(x[1] - x[0], 1), resolution=alpha_resolution,
            sample_range=alpha_range)
    ES = gdr.Entropy(mode="vector", resolution=alpha_resolution, sample_range=alpha_range,
            normalized=False)
    BC = gdr.BettiCurve(resolution=alpha_resolution, sample_range=alpha_range)

    if rank == 0:
        print(f"Found {len(pos_list)} samples")

    avg, rem = divmod(len(pos_list), n_ranks)
    count = [avg+1 if r < rem else avg for r in range(n_ranks)]
    offset = [sum(count[:r]) for r in range(n_ranks)]

    print(f"{rank} Calculating summary curves")
    start, end = offset[rank], offset[rank]+count[rank]
    es, bc = calc_summary(pos_list[start:end], [ES, BC], boxsize=boxsize)
    norm = np.trapz(np.abs(es), alpha, axis=-1)
    es = es / np.expand_dims(norm, 2)
    bc = bc / boxsize**3

    print(f"{rank} Scaling and interpolating ES curves")
    es_scaled = np.zeros((len(es), 3, scaled_resolution))
    bc_scaled = np.zeros_like(es_scaled)
    for i in range(len(es)):
        lbar = lbar_vals[start+i]
        interp_es = interp1d(alpha/lbar, es[i], axis=-1, bounds_error=False, fill_value=0, assume_sorted=True)
        es_scaled[i] = interp_es(alpha_scaled)
        interp_bc = interp1d(alpha/lbar, bc[i], axis=-1, bounds_error=False, fill_value=0, assume_sorted=True)
        bc_scaled[i] = interp_bc(alpha_scaled) * lbar**3
    norm_scaled = np.trapz(np.abs(es_scaled), alpha_scaled, axis=-1)
    es_scaled = es_scaled / np.expand_dims(norm_scaled, 2)

    comm.Barrier()
    if rank == 0:
        print("Collecting data")

    es = comm.gather(es)
    bc = comm.gather(bc)
    es_scaled = comm.gather(es_scaled)
    bc_scaled = comm.gather(bc_scaled)

    if rank == 0:
        name = fname.split('/')[-1]
        mock_type = name.split('_')[0]
        save_file = f"topology_summaries/{mock_type}/es_{name}"
        print(f"Saving data to {save_file}")

        es = np.vstack(es)
        bc = np.vstack(bc)
        es_scaled = np.vstack(es_scaled)
        bc_scaled = np.vstack(bc_scaled)

        np.savez(save_file, alpha=alpha, es=es, bc=bc,
                alpha_scaled=alpha_scaled, es_scaled=es_scaled, bc_scaled=bc_scaled,
                lbar=lbar_vals, boxsize=boxsize)


main()
