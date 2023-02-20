import sys
import glob
import numpy as np
import pandas as pd
import h5py


def read_rockstar(fname, fields=None):
    a = 1
    boxsize = 0
    h = 0.67
    with open(fname) as f:
        col_names = f.readline().lstrip('#').split()
        for i in range(10):
            line = f.readline()
            if line.startswith("#a"):
                a = float(line.split()[-1])
            elif line.startswith("#Box"):
                boxsize = float(line.split()[2])
            elif line.startswith("#Om"):
                h = float(line.split()[-1])
    redshift = 1/a - 1

    if fields=None:
        use_cols = ["Mvir", "M200c", "M200b", "X", "Y", "Z"]
    else:
        use_cols = fields

    data = pd.read_csv(fname, comment='#', sep=' ', names=col_names, usecols=use_cols, dtype='f4')

    return redshift, boxsize, h, data



def main():
    data_dir = sys.argv[1]
    save_dir = sys.argv[2]

    snap_num = 99
    digits = 2

    catalogs = glob.glob(data_dir + "/*/Rockstar/")
    print(len(catalogs), "catalogs found")

    for i, catalog in enumerate(catalogs[10:]):
        fname = catalog + f"out_{snap_num}.list"
        print(i, fname)
        redshift, boxsize, _, data = read_rockstar(fname)
        print(f"\t{len(data)} halos found")
        sim_name = fname.split('/')[-3]
        save_name = save_dir + '/' + sim_name + f"/rockstar_{snap_num:0{digits}}.hdf5"
        print("\tSaving to", save_name)
        with h5py.File(save_name, 'w') as f:
            f.attrs["redshift"] = redshift
            f.attrs["boxsize"] = boxsize
            f.attrs["Nhalos"] = len(data)
            for field in data.columns:
                if field in ["X", "Y", "Z"]:
                    continue
                f.create_dataset(field, data=data[field].values)
            f.create_dataset("pos", data=data[["X", "Y", "Z"]].values)



if __name__ == "__main__":
    main()
