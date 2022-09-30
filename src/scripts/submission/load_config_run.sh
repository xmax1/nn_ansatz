#!/bin/bash

submission_path=/home/energy/amawi/projects/nn_ansatz/src/scripts/submission/demo_list.sh

ngpus=(4 4 4)
paths=(/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/final0701/14el/new_baseline/kfac_1lr-3_1d-4_1nc-4_m2048_el14_s128_p32_l3_det1/run618506/config1.pk \
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/final0701/14el/new_baseline/kfac_1lr-3_1d-4_1nc-4_m2048_el14_s128_p32_l3_det1/run832591/config1.pk \
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/final0701/14el/new_baseline/kfac_1lr-3_1d-4_1nc-4_m2048_el14_s128_p32_l3_det1/run727015/config1.pk)
 
for i in "${!ngpus[@]}"
do
    ngpu=${ngpus[i]}
    path="-p ${paths[i]}"
    echo $path
    sbatch --gres=gpu:RTX3090:$ngpu --job-name=p $submission_path $path
    echo ngpu $ngpu
done

