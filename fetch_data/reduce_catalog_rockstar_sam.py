import os
import sys
import glob
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
            g_fields = lambda x: x not in ["sfrave20myr", "sfrave100myr", "sfrave1gyr", "tmerge",
                                            "tmajmerge", "mu_merge", "mass_outflow_rate",
                                            "metal_outflow_rate", "maccdot", "maccdot_radio",
                                            "r_fric"]
        g_dtypes = {"halo_index": 'u4', "birthhaloid": 'u4', "roothaloid": 'u4', "sat_type": 'u4'}
        for colname in g_colnames:
            if colname not in g_dtypes.keys():
                g_dtypes[colname] = 'f4'

        h_colnames = ['halo_index', 'halo_id', 'roothaloid', 'orig_halo_ID', 'redshift', 'm_vir',
                    'c_nfw', 'spin', 'm_hot', 'mstar_diffuse', 'mass_ejected', 'mcooldot',
                    'maccdot_pristine', 'maccdot_reaccrete', 'maccdot_metal_reaccrete',
                    'maccdot_metal', 'mdot_eject', 'mdot_metal_eject', 'maccdot_radio', 'Metal_hot',
                    'Metal_ejected', 'snap_num']
        if h_fields is None:
            h_fields = lambda x: x not in ["snap_num"]
        h_dtypes = {"halo_index": 'u4', "halo_id": 'u4', "roothaloid": 'u4', "orig_halo_ID": 'u4'}
        for colname in h_dtypes:
            if colname not in h_dtypes.keys():
                h_dtypes[colname] = 'f4'

        halo_df = pd.read_csv(halo_data, sep=' ', skiprows=len(h_colnames), names=h_colnames,
                            usecols=h_fields, engine=engine, dtype=h_dtypes)
        halo_dfs.append(halo_df)

        gal_df = pd.read_csv(gal_data, sep=' ', skiprows=len(g_colnames), names=g_colnames,
                            usecols=g_fields, engine=engine, dtype=g_dtypes)
        gal_dfs.append(gal_df)

    halo_data = pd.concat(halo_dfs)
    gal_data = pd.concat(gal_dfs)
    print("\tFound ", len(halo_data), "halos")
    print("\tFound ", len(gal_data), "galaxies")

    return halo_data, gal_data


def reduce_sam_catalog(sam_dir, rockstar_dir, snaps, save_dir=None):
    rockstar_snaps = [rockstar_dir + f"/out_{snap}.list" for snap in snaps]
    snap_z_vals = []
    rockstar_halos = []

    # First attempt to read rockstar file
    for snap in rockstar_snaps:
        z, boxsize, h, rockstar_data = read_rockstar(snap, fields=["Mvir", "X", "Y", "Z"])
        snap_z_vals.append(z)
        rockstar_halos.append(rockstar_data)

    print(boxsize)

    haloprop, galprop = read_sam(sam_dir, g_fields=["redshift", "sat_type", "mstar", "sfr",
        "x_position", "y_position", "z_position"], h_fields=[])

    haloprop_z = []
    galprop_z = []
    for z in snap_z_vals:
        selection = np.isclose(galprop["redshift"].values, z, atol=1e-3)
        galprop_z.append(galprop.drop(columns="redshift").take(np.nonzero(selection)[0]))
        galprop = galprop.take(np.nonzero(~selection)[0])

    for i in range(len(snaps)):
        print(snap_z_vals[i], len(rockstar_halos[i]), "halos,", len(galprop_z[i]), "galaxies")
        
        catalog_reduced = f"rockstar_sam_tab_{snaps[i]:03}_reduced.hdf5"
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            catalog_reduced = save_dir + '/' + catalog_reduced
        print("saving catalog to", catalog_reduced)

        halo_pos = rockstar_halos[i][["X", "Y", "Z"]].values
        gal_pos = galprop_z[i][["x_position", "y_position", "z_position"]].values * h
        # make sure positions are inside box
        gal_pos[gal_pos >= boxsize] -= boxsize
        gal_pos[gal_pos < 0] += boxsize
        gal_mass = galprop_z[i]["mstar"].values * h * 1e9
        gal_central = galprop_z[i]["sat_type"].values == 0

        with h5py.File(catalog_reduced, 'w') as f:
            f.attrs["boxsize"] = boxsize
            f.attrs["redshift"] = snap_z_vals[i]
            f.create_dataset("HaloPos", data=halo_pos)
            f.create_dataset("HaloMass", data=rockstar_halos[i]["Mvir"].values)
            f.create_dataset("SubhaloPos", data=gal_pos)
            f.create_dataset("SubhaloStMass", data=gal_mass)
            f.create_dataset("SubhaloSFR", data=galprop_z[i]["sfr"].values)
            f.create_dataset("SubhaloCentral", data=gal_central)
            
            
def main():
    # locations of sam and rockstar files
    sam_base = "/home/jovyan/Data/SCSAM"
    rockstar_base = "/home/jovyan/Data/Rockstar/CAMELS-SAM"
    
    save_dir = "/home/jovyan/home/camels-sam"
    
    one_p_sims = {"Asn1x0p25": "1P_1", "Asn1x4p0": "1P_2",
                  "Asn2x0p25": "1P_3", "Asn2x4p0": "1P_4",
                  "Aagn1x0p25": "1P_5", "Aagn1x4p0": "1P_6"}
    
    def one_p_params(fid_params, sim_name):
        if sim_name == "1P_1":
            return fid_params * np.array([1,1,0.25,1,1])
        elif sim_name == "1P_2":
            return fid_params * np.array([1,1,4,1,1])
        elif sim_name == "1P_3":
            return fid_params - np.array([0,0,0,2,0])
        elif sim_name == "1P_4":
            return fid_params + np.array([0,0,0,2,0])
        elif sim_name == "1P_5":
            return fid_params * np.array([1,1,1,1,0.25])
        elif sim_name == "1P_6":
            return fid_params * np.array([1,1,1,1,4])
        raise ValueError        
    
    # extract data for all simulations at given snapshots
    snaps = sys.argv[1:]
    snaps = list(map(int, snaps))
    
    sims = glob.glob(sam_base + "/*_*/")
    sims.remove(sam_base+"/LH_1000/")  # no data in these dirs
    sims.remove(sam_base+"/CV_5/")
    
    print(f"Found {len(sims)} sims")
    
    params_all = {}
    
    for sim in sims:
        sim_name = sim.rstrip('/').split('/')[-1]
        params = np.genfromtxt(sim + "/CosmoAstro_params.txt")[:-1]
        params_all[sim_name] = params
        print(sim_name)
        rockstar_dir = rockstar_base + '/' + sim_name + "/Rockstar"
        
        if "CV" in sim_name:
            sam_dir = sam_base + '/' + sim_name + "/fid-sc-sam"
            reduce_sam_catalog(sam_dir, rockstar_dir, snaps, save_dir=save_dir+'/'+sim_name)
            
            sims_1p = glob.glob(sim + "A*-sc-sam")
            print(f"Found {len(sims_1p)} 1P sims")
            for sim_1p in sims_1p:
                sim_name_1p = sim_1p.rstrip('/').split('/')[-1].rstrip("-sc-sam")
                sim_name_1p = one_p_sims[sim_name_1p]
                params_1p = one_p_params(params, sim_name_1p)
                if sim_name_1p not in params_all.keys():
                    params_all[sim_name_1p] = params_1p
                print(sim_name_1p)
                
                reduce_sam_catalog(sim_1p, rockstar_dir, snaps, save_dir=save_dir+'/'+sim_name+'/'+sim_name_1p)
            
        elif "LH" in sim_name:
            sam_dir = sam_base + '/' + sim_name + "/sc-sam"
            reduce_sam_catalog(sam_dir, rockstar_dir, snaps, save_dir=save_dir+'/'+sim_name)
    
    with open(save_dir + "/CosmoAstro_params.txt", 'w') as f:
        for sim in sorted(params_all.keys()):
            params = params_all[sim]
            f.write(f"{sim}\t{params[0]:.5f}\t{params[1]:.5f}\t{params[2]:.5f}\t{params[3]:.5f}\t{params[4]:.5f}\n")
            
    print("Done.")
            

if __name__ == "__main__":
    main()
