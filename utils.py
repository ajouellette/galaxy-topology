import glob
import h5py
import numpy as np
import illustris_python as tng


# Need a unified interface to access:
#  simulation box size
#  halo positions and masses
#  galaxy positions, stellar masses, SFR, central/satellite


def load_illustris(sim_name, snap_num=-1):
    """Load an Illustris or TNG halo/subhalo catalog.

    Additionally provides access to: subhalo DM fraction and match between subhalo
    and mass of surrounding halo.
    Catalogs based on FoF/Subfind.
    Data is split between many HDF5 files stored in groups_*/
    """

    if snap_num == -1:
        snaps = glob.glob(sim_name + "/groups_*")
        if len(snaps) == 0:
            raise ValueError("No group directories found in "+sim_name)
        snaps.sort()
        snap_num = int(snaps[-1].split('_')[-1])
    header = tng.groupcat.loadHeader(sim_name, snap_num)
    halo_data = tng.groupcat.loadHalos(sim_name, snap_num,
                ["GroupPos", "Group_M_Crit200", "Group_M_TopHat200", "GroupFirstSub"])
    subhalo_data = tng.groupcat.loadSubhalos(sim_name, snap_num,
            ["SubhaloPos", "SubhaloMassType", "SubhaloSFR", "SubhaloGroupNr"])
    halo_mass = halo_data["Group_M_TopHat200"] * 1e10
    sub_dm_frac = subhalo_data["SubhaloMassType"][:,1] / np.sum(subhalo_data["SubhaloMassType"], axis=1)
    sub_central = np.isin(np.arange(header["Nsubgroups_Total"]), halo_data["GroupFirstSub"])

    data = {"boxsize": header["BoxSize"] / 1e3,
            "HaloPos": halo_data["GroupPos"] / 1e3,
            "HaloMass": halo_mass,
            "SubhaloPos": subhalo_data["SubhaloPos"] / 1e3,
            "SubhaloStMass": subhalo_data["SubhaloMassType"][:,4] * 1e10,
            "SubhaloDmFrac": sub_dm_frac,
            "SubhaloSFR": subhalo_data["SubhaloSFR"],
            "SubhaloCentral": sub_central,
            "SubhaloHaloMass": halo_mass[subhalo_data["SubhaloGroupNr"]]}

    return data


def load_camels(sim_name, snap_num=-1):
    """Load a CAMELS-IllustrisTNG halo/subhalo catalog.

    Data is generally in the same format as the original Illustris/TNG catalogs,
    but is stored in only one HDF5 file.
    """

    if snap_num == -1:
        files = glob.glob(sim_name + "/fof_subhalo_tab_*.hdf5")
        if len(files) == 0:
            raise ValueError("No snapshot files found in " + sim_name)
        files.sort()
        fname = files[-1]
    else:
        fname = sim_name + f"/fof_subhalo_tab_{snap_num:03}.hdf5"
    data = {}
    with h5py.File(fname) as f:
        halo_mass = f["Group/Group_M_TopHat200"][:] * 1e10
        sub_mass = f["Subhalo/SubhaloMassType"][:] * 1e10
        dm_frac = sub_mass[:,1] / np.sum(sub_mass, axis=1)
        sub_central = np.isin(np.arange(f["Header"].attrs["Nsubgroups_Total"]),
                            f["Group/GroupFirstSub"][:])

        data["boxsize"] = f["Header"].attrs["BoxSize"] / 1e3
        data["HaloPos"] = f["Group/GroupPos"][:] / 1e3
        data["HaloMass"] = halo_mass
        data["SubhaloPos"] = f["Subhalo/SubhaloPos"][:] / 1e3
        data["SubhaloStMass"] = sub_mass[:,4]
        data["SubhaloDmFrac"] = dm_frac
        data["SubhaloSFR"] = f["Subhalo/SubhaloSFR"][:]
        data["SubhaloHaloMass"] = halo_mass[f["Subhalo/SubhaloGroupNr"][:]]
        data["SubhaloCentral"] = sub_central

    return data


def load_sam(sim_name, snap_num=-1):
    """Load a CAMELS-SAM halo/galaxy catalog.

    Catalogs based on Rockstar/SC-SAM.
    """

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
        data["HaloMass"] = f["Mvir"][:]
    with h5py.File(sam_fname) as f:
        data["SubhaloPos"] = f["galprop/pos"][:] * h
        data["SubhaloPos"][data["SubhaloPos"] < 0] += data["boxsize"]
        data["SubhaloPos"][data["SubhaloPos"] >= data["boxsize"]] -= data["boxsize"]
        data["SubhaloStMass"] = f["galprop/mstar"][:] * 1e9 * h
        data["SubhaloSFR"] = f["galprop/sfr"][:]
        data["SubhaloCentral"] = f["galprop/sat_type"][:] == 0

    return data
