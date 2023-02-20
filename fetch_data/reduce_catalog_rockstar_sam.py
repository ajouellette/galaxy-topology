import os
import sys
import numpy as np
import h5py
import pandas as pd
from convert_rockstar_data import read_rockstar
from convert_sam_data import read_sam_data


def read_sam(root_dir, g_fields=None, h_fields=None):
    subvols = glob.glob(f"{root_dir}/*_*_*/")
    if len(subvols) == 0:
        raise ValueError("No subvols found in "+root_dir)

    print('\t', len(subvols), "sub-volumes found")

    engine='c'

    halo_dfs = []
    gal_dfs = []
    for subvol in subvols:
        halo_data = subvol + "haloprop_0-99.dat"
        gal_data = subvol + "galprop_0-99.dat"

        g_colnames = ['halo_index', 'birthhaloid', 'roothaloid', 'redshift', 'sat_type',
                  'mhalo', 'm_strip', 'rhalo', 'mstar', 'mbulge', 'mstar_merge', 'v_disk',
                  'sigma_bulge', 'r_disk', 'r_bulge', 'mcold', 'mHI', 'mH2', 'mHII', 'Metal_star',
                  "Metal_cold", 'sfr', 'sfrave20myr', 'sfrave100myr', 'sfrave1gyr',
                  'mass_outflow_rate', 'metal_outflow_rate', 'mBH', 'maccdot', 'maccdot_radio',
                  'tmerge', 'tmajmerge', 'mu_merge', 't_sat', 'r_fric', 'x_position',
                  'y_position', 'z_position', 'vx', 'vy', 'vz']
        if g_fields is None:
            g_fields = lambda x: x not in ["sfrave20myr", "sfrave100myr", "sfrave1gyr", "tmerge", "tmajmerge",
                                           "mu_merge", "mass_outflow_rate", "metal_outflow_rate", "maccdot",
                                           "maccdot_radio", "r_fric"]
        g_dtypes = {"halo_index": 'u4', "birthhaloid": 'u4', "roothaloid": 'u4', "sat_type": 'u4'}
        for colname in g_colnames:
            if colname not in g_dtypes.keys():
                g_dtypes[colname] = 'f4'

        h_colnames = ['halo_index', 'halo_id', 'roothaloid', 'orig_halo_ID', 'redshift', 'm_vir', 'c_nfw',
                  'spin', 'm_hot', 'mstar_diffuse', 'mass_ejected', 'mcooldot',
                  'maccdot_pristine', 'maccdot_reaccrete', 'maccdot_metal_reaccrete',
                  'maccdot_metal', 'mdot_eject', 'mdot_metal_eject', 'maccdot_radio',
                  'Metal_hot', 'Metal_ejected', 'snap_num']
        if h_fields is None:
            h_fields = lambda x: x not in ["snap_num"]
        h_dtypes = {"halo_index": 'u4', "halo_id": 'u4', "roothaloid": 'u4', "orig_halo_ID": 'u4'}
        for colname in h_dtypes:
            if colname not in h_dtypes.keys():
                h_dtypes[colname] = 'f4'

        halo_df = pd.read_csv(halo_data, sep=' ', skiprows=len(h_colnames), names=h_colnames, usecols=h_fields,
                              engine=engine, dtype=h_dtypes)
        halo_dfs.append(halo_df)

        gal_df = pd.read_csv(gal_data, sep=' ', skiprows=len(g_colnames), names=g_colnames, usecols=g_fields,
                             engine=engine, dtype=g_dtypes)
        gal_dfs.append(gal_df)

    halo_data = pd.concat(halo_dfs)
    gal_data = pd.concat(gal_dfs)
    print("\tFound ", len(halo_data), "halos")
    print("\tFound ", len(gal_data), "galaxies")

    return halo_data, gal_data


if __name__ == "__main__":
def main():
    rockstar_dir = sys.argv[1].rstrip('/')
    sam_dir = sys.argv[2].rstrip('/')
    snaps = sys.argv[3:]
    if isinstance(snaps, list):
        snaps = map(int, snaps)
    else:
        snaps = [int(snaps),]

    if not os.path.isdir(rockstar_dir):
        raise ValueError(f"'{rockstar_dir}' is not a directory")
    if not os.path.isdir(sam_dir):
        raise ValueError(f"'{sam_dir}' is not a directory")

    rockstar_snaps = [rockstar_dir + f"/out_{snap}.list" for snap in snaps]
    snap_z_vals = []
    rockstar_halos = []

    # First attempt to read rockstar file
    for snap in rockstar_snaps:
        z, boxsize, h, rockstar_data = read_rockstar(rockstar, fields=["Mvir", "X", "Y", "Z"])
        snap_z_vals.append(z)
        rockstar_halos.append(rockstar_data)

    haloprop, galprop = read_sam(sam_dir)

    haloprop_z = []
    galprop_z = []




if __name__ == "__main__":
    main()
