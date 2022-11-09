#!/usr/bin/env python3

import os
import subprocess
import sys

temp_batch = ".batch.txt"


def main():
    help_msg = f"""Usage: {sys.argv[0]} SuiteName SimSet files.txt
    Submit a Globus transfer task to download specified files from the CAMELS dataset.
    files.txt is a list of files to download from each simulation in SuiteName/SimSet.
    """

    if len(sys.argv) != 4:
        print(help_msg)
        sys.exit()

    suite = sys.argv[1]
    sim_set = sys.argv[2]
    template = sys.argv[3]

    command = f"globus ls $(globus bookmark show camels)/Sims/{suite} | grep {sim_set}"
    result = subprocess.run(command, capture_output=True, shell=True, text=True)

    sims = result.stdout.split()
    print(f"Found {len(sims)} simulations in {suite}/{sim_set}")

    with open(template, "r") as f:
        files = f.read().split()

    print(f"Found {len(files)} files in {template}")
    print(f"{len(files) * len(sims)} files to download")
    print("Generating Globus batch file...")

    batch = ""
    for i, sim in enumerate(sims):
        for j, fname in enumerate(files):
            batch = batch + f"{sim.rstrip('/')}/{fname} {sim.rstrip('/')}/{fname}\n"
        if i < len(sims) - 1:
            batch = batch + "\n"

    with open(temp_batch, "w") as f:
        f.write(batch.strip())

    response = "-"
    while response not in ["y", "n", ""]:
        response = input("Confirm batch file [Y/n]? ").lower()

    confirm_batch = False
    if response == "y" or response == "":
        confirm_batch = True

    if confirm_batch:
        subprocess.run(["less", "-N", temp_batch])

    response = "-"
    while response not in ["y", "n", ""]:
        response = input("Start transfer [Y/n]? ").lower()

    confirm_transfer = False
    if response == "y" or response == "":
        confirm_transfer = True

    if confirm_transfer:
        print("Submitting transfer task...")
        command = f"globus transfer --skip-source-errors --batch {temp_batch}"
        command += f" $(globus bookmark show camels)/Sims/{suite}/"
        command += f" $(globus bookmark show my_caps)/topology/camels/{suite}/"

        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
