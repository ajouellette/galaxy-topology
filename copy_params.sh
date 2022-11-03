#!/bin/bash

data_dir="$1"
save_dir="$2"

for f in $(find $data_dir -name CosmoAstro_params.txt)
do
	sim_name=$(basename $(dirname $f))
	cp -nv $f $save_dir/$sim_name/
done
