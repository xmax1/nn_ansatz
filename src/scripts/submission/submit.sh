#!/bin/bash

submission_path=/home/energy/amawi/projects/nn_ansatz/src/scripts/submission/demo.sh

if [[ $2 -eq 0 ]]; then
    ngpu=1
else
    ngpu=$2
fi

if [ "$1" == "twoloop" ]  # the space is needed because [ is a test ]
then
    echo Calling twoloop submission

    myArray=(cos 2cos 3cos 4cos 2cos+2sin 3cos+3sin)
    myArray2=(tanh cos)
    # myArray=(16 32 64 128)
    # myArray2=(4 8 16 32)
    for hypam in "${myArray[@]}" 
    do 
        for hypam2 in "${myArray2[@]}"
        do
            cmd="-s HEG \
                -sim 1 1 1 \
                -nw 1024 \
                -n_sh 128 \
                -n_ph 32 \
                -nl 3 \
                -n_det 16 \
                -orb real_plane_waves \
                -n_el 7 \
                -inact $hypam \
                -act $hypam2 \
                -dp 1 \
                -name 2611/find_act \
                -n_it 100000 "
            sbatch --gres=gpu:RTX3090:$ngpu --job-name=actsweep $submission_path $cmd
            echo $hypam $hypam2
            sleep 20
        done
    done
fi

if [ "$1" == "oneloop" ]; then 
    echo Calling single loop
    myArray=(1 50 100) 
    # myArray=(512 1024 2048 4096)
    # myArray=(1 2 3 4 5)
    # myArray=(0 1 2 3 4 5)
    # myArray=(7 19 27 33 57)
    # ngpus=(1 1 1 1 1)
    for i in "${!myArray[@]}"
    # for hypam in "${myArray[@]}" 
    do
        hypam=${myArray[i]}
        # ngpu=${ngpus[i]}
        # n_sh=$(( $hypam*16 ))
        # n_ph=$(( $hypam*8 ))
        cmd="-s HEG \
        -nw 2048 \
        -n_sh 128 \
        -n_ph 32 \
        -nl 3 \
        -n_det 1 \
        -orb real_plane_waves \
        -n_el 14 \
        -n_up 7 \
        -inact 5cos+5sin+19kpoints \
        -dp $hypam \
        -name the_reup \
        -n_it 100000 \
        -backflow_coords True \
        -psplit_spins True \
        -jastrow False \
        -lr 0.001 \
        -cl 10 \
        -ta 0.5"
        sbatch --gres=gpu:RTX3090:$ngpu --job-name=e14dp$hypam $submission_path $cmd
        echo $cmd
        echo ngpu $ngpu
    done
fi

if [ "$1" == "single" ]; then 
    echo Calling single submission
    cmd="-s HEG \
        -nw 2048 \
        -n_sh 128 \
        -n_ph 32 \
        -nl 3 \
        -n_det 1 \
        -orb real_plane_waves \
        -n_el 7 \
        -n_up 7 \
        -inact 5cos+5sin+19kpoints \
        -dp 1 \
        -name logdetsum/test/1 \
        -n_it 100000 \
        -backflow_coords True \
        -jastrow False \
        -psplit_spins True \
        -lr 0.001 \
        -cl 20 \
        -ta 0.5"

    echo $cmd
    sbatch --gres=gpu:RTX3090:$ngpu --job-name=logdettest $submission_path $cmd
fi


# HEG
# 7 el
# 1024 128 32 3 7 0 1gpu tick
# 2048 128 32 3 7 0 1gpu tick
# 1024 128 32 3 7 7 1gpu OOM 512 tick 256 tick
# 1024 128 32 3 7 8 1gpu OOM 512 256 
# 2048 128 32 3 7 7 2gpu OOM 512 tick 256 tick

# 1024 128 32 2l 7el 1in tick
# 2048 64 16 2l 7el 1in tick

# 14 el para
# 1024 128 32 3 1gpu 

# 19 el
# 1024 128 32 2 19el 34GB
# 1024 64 16 4 el 34GB
# 1024 64 16 3 19el 
# 1024 64 16 2 19el 1gpu 
# 1024 64 16 2 19el 5in 24GB
# 2048 64 16 2 19el 1in 34GB
