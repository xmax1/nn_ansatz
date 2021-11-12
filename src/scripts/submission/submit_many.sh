#!/bin/bash

submission_path=/home/energy/amawi/projects/nn_ansatz/src/scripts/submission/demo.sh

myArray=(bowl sin cos)
myArray2=(tanh sin)
for hypam in "${myArray[@]}" 
do 
    for hypam2 in "${myArray2[@]}"
    do
        cmd="-s HEG \
            -sim 1 1 1 \
            -nw 512 \
            -n_sh 64 \
            -n_ph 16 \
            -nl 2 \
            -n_det 1 \
            -orb real_plane_waves \
            -n_el 7 \
            -inact $hypam \
            -act $hypam2 \
            -dp 1 \
            -name 1211/inact_act3 \
            -n_it 100000"
        sbatch $submission_path $cmd
        sleep 5
    done
done
