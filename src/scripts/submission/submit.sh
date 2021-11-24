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
    myArray2=(tanh sin cos silu)
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
                -name 1811/find_act \
                -n_it 10000 \
                -lr 0.001"
            sbatch --gres=gpu:RTX3090:$ngpu --job-name=actsweep $submission_path $cmd
            echo $hypam $hypam2
            sleep 20
        done
    done
fi

if [ "$1" == "oneloop" ]; then 
    echo Calling single loop
    myArray=(0.5 1 2 5 10 20 50 100)
    # myArray=(1 2 5 10)
    # myArray=(512 1024 )
    # myArray=(2048 )
    # myArray=(4096 )
    # myArray=(1 2 4 8 16)
    # myArray=(1 2 4 8 16)
    # myArray=(4 8 )
    for hypam in "${myArray[@]}" 
    do
        # n_sh=$(( $hypam*16 ))
        # n_ph=$(( $hypam*4 ))
        cmd="-s HEG \
            -sim 1 1 1 \
            -nw 4096 \
            -n_sh 128 \
            -n_ph 32 \
            -nl 3 \
            -n_det 1 \
            -orb real_plane_waves \
            -n_el 7 \
            -dp $hypam \
            -name 2111/find_jastrow_sin \
            --jastrow \
            -n_it 100000 \
            -lr 0.001"
        sbatch --gres=gpu:RTX3090:$ngpu --job-name=n_params $submission_path $cmd
        echo $cmd
        sleep 3
    done
fi

if [ "$1" == "single" ]; then 
    echo Calling single submission

    cmd="-s HEG \
        -sim 1 1 1 \
        -nw 1028 \
        -n_sh 128 \
        -n_ph 32 \
        -nl 3 \
        -n_det 1 \
        -orb real_plane_waves \
        -n_el 7 \
        -dp 1 \
        -name 2411/kfac_sum_not_mean \
        -n_it 10000 \
        -lr 0.001"
    sbatch --gres=gpu:RTX3090:$ngpu --job-name=para_el7 $submission_path $cmd
fi


# HEG
# 7 el
# 1024 128 32 3 7 0 1gpu tick
# 2048 128 32 3 7 0 1gpu tick
# 1024 128 32 3 7 7 1gpu OOM 512 tick 256 tick
# 1024 128 32 3 7 8 1gpu OOM 512 256 
# 2048 128 32 3 7 7 2gpu OOM 512 tick 256 tick

# 14 el para
# 1024 128 32 3 1gpu 

# 19 el
# 1024 64 16 19 el 2gpu tick 
# 1024 64 16 2 19el 1gpu 
# 
