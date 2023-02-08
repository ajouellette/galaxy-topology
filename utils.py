import glob
import h5py
import numpy as np
import illustris_python as tng


def load_camels(sim_name, snap_num=-1):
    """Load a CAMELS-IllustrisTNG sim."""

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
            "HaloPos": halo_data["GroupPos"] / 1e3,
            "HaloMass": halo_data["Group_M_Crit200"] * 1e10,
            "SubhaloPos": subhalo_data["SubhaloPos"] / 1e3,
            "SubhaloMass": subhalo_data["SubhaloMassType"] * 1e10,
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
