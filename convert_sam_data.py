import os
import sys
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py


def read_sam_data(root_dir, snap_num, max_snap_num=99):
    subvols = glob.glob(f"{root_dir}/*_*_*/")
    print(len(subvols), "sub-volumes found")

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
        g_usecols = lambda x: x not in ["sfrave20myr", "sfrave100myr", "sfrave1gyr"]
        g_dtypes = defaultdict(lambda: 'f4', {"halo_index": 'u4', "birthhaloid": 'u4', "roothaloid": 'u4', "sat_type": 'u4'})

        h_colnames = ['halo_index', 'halo_id', 'roothaloid', 'orig_halo_ID', 'redshift', 'm_vir', 'c_nfw',
                  'spin', 'm_hot', 'mstar_diffuse', 'mass_ejected', 'mcooldot',
                  'maccdot_pristine', 'maccdot_reaccrete', 'maccdot_metal_reaccrete',
                  'maccdot_metal', 'mdot_eject', 'mdot_metal_eject', 'maccdot_radio',
                  'Metal_hot', 'Metal_ejected', 'snap_num']
        h_usecols = lambda x: x not in ["snap_num"]
        h_dtypes = defaultdict(lambda: 'f4', {"halo_index": 'u4', "halo_id": 'u4', "roothaloid": 'u4', "orig_halo_ID": 'u4'})

        halo_df = pd.read_csv(halo_data, sep=' ', skiprows=len(h_colnames), names=h_colnames, usecols=h_usecols, engine=engine, dtype=h_dtypes)
        redshifts = np.unique(halo_df["redshift"])
        snap_z = redshifts[max_snap_num - snap_num]
        mask = halo_df["redshift"] == snap_z
        halo_df.drop(columns="redshift", inplace=True)
        halo_dfs.append(halo_df[mask])

        gal_df = pd.read_csv(gal_data, sep=' ', skiprows=len(g_colnames), names=g_colnames, usecols=g_usecols, engine=engine, dtype=g_dtypes)
        mask = gal_df["redshift"] == snap_z
        gal_df.drop(columns="redshift", inplace=True)
        gal_dfs.append(gal_df[mask])

    return snap_z, pd.concat(halo_dfs), pd.concat(gal_dfs)


def save_snapshot(root_dir, save_dir, snap_num, fname_base="sc_sam_tab", max_snap_num=99):

    redshift, halo_data, gal_data = read_sam_data(root_dir, snap_num, max_snap_num=max_snap_num)

    # save data
    digits = len(str(max_snap_num))
    fname = f"{save_dir}/{fname_base}_{snap_num:0{digits}}.hdf5"
    print("saving to", fname)
    with h5py.File(fname, 'w') as f:
        f.attrs["redshift"] = redshift
        f.attrs["Nhalos"] = len(halo_data)
        f.attrs["Ngalaxies"] = len(gal_data)
        for field in halo_data.columns:
            f.create_dataset("haloprop/"+field, data=halo_data[field].values)
        for field in gal_data.columns:
            if field in ["x_position", "y_position", "z_position", "vx", "vy", "vz"]:
                continue
            f.create_dataset("galprop/"+field, data=gal_data[field].values)
        f.create_dataset("galprop/pos", data=gal_data[["x_position", "y_position", "z_position"]].values)
        f.create_dataset("galprop/vel", data=gal_data[["vx", "vy", "vz"]].values)


def main():

    root_dir = sys.argv[1]
    snap_num = int(sys.argv[2])

    save_snapshot(root_dir, root_dir, snap_num)


if __name__ == "__main__":
    main()
