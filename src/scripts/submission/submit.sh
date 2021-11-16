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

    myArray=(bowl sin cos bowl+cos bowl+sin sin+cos bowl+sin+cos bowl+2sin+2cos)
    myArray2=(tanh sin cos)
    # myArray=(16 32 64 128)
    # myArray2=(4 8 16 32)
    for hypam in "${myArray[@]}" 
    do 
        for hypam2 in "${myArray2[@]}"
        do
            cmd="-s HEG \
                -sim 1 1 1 \
                -nw 1024 \
                -n_sh 64 \
                -n_ph 16 \
                -nl 3 \
                -n_det 1 \
                -orb real_plane_waves \
                -n_el 19 \
                -inact $hypam \
                -act $hypam2 \
                -dp 1 \
                -name 1311/19el_inact_act_sweep_1024 \
                -n_it 10000 \
                -lr 0.001"
            sbatch --gres=gpu:RTX3090:$ngpu $submission_path $cmd
            echo $hypam $hypam2
            sleep 15
        done
    done
fi
if [ "$1" == "oneloop" ]
then 
    echo Calling single loop
    myArray=(1 2 3 4 5 10)
    for hypam in "${myArray[@]}" 
    do
        #n_sh=$(( $hypam*32 ))
        #n_ph=$(( $hypam*8 ))
        cmd="-s HEG \
            -sim 1 1 1 \
            -nw 1024 \
            -n_sh 128 \
            -n_ph 32 \
            -nl 3 \
            -n_det 1 \
            -orb real_plane_waves \
            -n_el 7 \
            -inact cos \
            -act cos \
            -dp $hypam \
            -name 1511/el19_dp_sweep \
            -n_it 10000  \
            -lr 0.001"
        sbatch --gres=gpu:RTX3090:$ngpu $submission_path $cmd
        echo $cmd
        sleep 15
    done
fi

if [ "$1" == "single" ]
then 
    echo Calling single submission

    cmd="-s HEG \
        -sim 1 1 1 \
        -nw 1024 \
        -n_sh 128 \
        -n_ph 32 \
        -nl 3 \
        -n_det 1 \
        -orb real_plane_waves \
        -n_el 7 \
        -inact cos \
        -act cos \
        -dp 1 \
        -name 1511/7el_opt_sweep \
        -n_it 100000 \
        --sweep"
    sbatch --gres=gpu:RTX3090:$ngpu $submission_path $cmd

fi




