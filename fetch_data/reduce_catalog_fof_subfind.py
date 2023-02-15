import os
import sys
import numpy as np
import h5py
import illustris_python as tng


halo_fields = ["GroupPos", "Group_M_TopHat200", "GroupFirstSub"]
subhalo_fields = ["SubhaloPos", "SubhaloMassType", "SubhaloGrNr", "SubhaloSFR"]


def main():
    catalog = sys.argv[1].rstrip('/')

    if os.path.isdir(catalog):
        print("Loading data from directory...")
        path = os.path.dirname(catalog)
        snap_num = int(os.path.basename(catalog).split('_')[-1])
        header = tng.groupcat.loadHeader(path, snap_num)
        halo_data = tng.groupcat.loadHalos(path, snap_num, halo_fields)
        subhalo_data = tng.groupcat.loadSubhalos(path, snap_num, subhalo_fields)
    elif os.path.isfile(catalog):
        print("Loading data from file...")
        with h5py.File(catalog) as f:
            header = dict(f["Header"].attrs.items())
            halo_data = {}
            for field in halo_fields:
                halo_data["count"] = header["Ngroups_Total"]
                halo_data[field] = f["Group"][field][:]
            subhalo_data = {}
            for field in subhalo_fields:
                subhalo_data["count"] = header["Nsubgroups_Total"]
                subhalo_data[field] = f["Subhalo"][field][:]
    else:
        raise ValueError(f"'{catalog}' is not a file or directory")

    boxsize = header["BoxSize"] / 1e3
    redshift = header["Redshift"]

    halo_mass = halo_data["Group_M_TopHat200"] * 1e10
    sub_mass = np.sum(subhalo_data["SubhaloMassType"], axis=1) * 1e10
    sub_dm_frac = subhalo_data["SubhaloMassType"][:,1] * 1e10 / sub_mass
    sub_central = np.isin(np.arange(subhalo_data["count"]), halo_data["GroupFirstSub"],
            assume_unique=True)


    catalog_reduced = f"{os.path.splitext(catalog)[0]}_reduced.hdf5"
    print("Saving reduced catalog to", catalog_reduced)

    with h5py.File(catalog_reduced, 'w') as f:
        f.attrs["boxsize"] = boxsize
        f.attrs["redshift"] = redshift
        f.create_dataset("HaloPos", data=halo_data["GroupPos"]/1e3)
        f.create_dataset("HaloMass", data=halo_mass)
        f.create_dataset("SubhaloPos", data=subhalo_data["SubhaloPos"]/1e3)
        f.create_dataset("SubhaloMass", data=sub_mass)
        f.create_dataset("SubhaloStMass", data=subhalo_data["SubhaloMassType"][:,4]*1e10)
        f.create_dataset("SubhaloDmFrac", data=sub_dm_frac)
        f.create_dataset("SubhaloSFR", data=subhalo_data["SubhaloSFR"])
        f.create_dataset("SubhaloCentral", data=sub_central)
        f.create_dataset("SubhaloHaloMass", data=halo_mass[subhalo_data["SubhaloGrNr"]])


if __name__ == "__main__":
    main()
