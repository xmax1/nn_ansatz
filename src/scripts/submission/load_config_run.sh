#!/bin/bash

submission_path=/home/energy/amawi/projects/nn_ansatz/src/scripts/submission/demo_list.sh

ngpus=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 4 4 4 4 2 1 1 1 1 1 1 1 1 1 1 1 1)
paths=(/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/psplit/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run442990/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/backflow/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run866044/config1.pk 
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run476218/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run683801/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run613269/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run687238/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run471041/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run322693/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run350643/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run856425/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/kpoints/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run279996/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/kpoints/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run160567/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/kpoints/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run211460/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/kpoints/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run306022/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/nwalkers/kfac_1lr-3_1d-4_1nc-4_m4096_el14_s32_p16_l2_det1/run54062/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/nwalkers/kfac_1lr-3_1d-4_1nc-4_m1024_el14_s32_p16_l2_det1/run225920/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/nwalkers/kfac_1lr-3_1d-4_1nc-4_m2048_el14_s32_p16_l2_det1/run601987/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/nwalkers/kfac_1lr-3_1d-4_1nc-4_m2048_el14_s32_p16_l2_det1/run620204/config1.pk 
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/np/kfac_1lr-3_1d-4_1nc-4_m512_el14_s128_p64_l2_det1/run853230/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/cl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run471095/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/cl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run425143/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/cl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run88208/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/cl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run365702/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/cl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run316978/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/cl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run250755/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/nl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l1_det1/run610414/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/nl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l1_det1/run957808/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/nl/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run310909/config1.pk  
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/kp_3ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run327452/config1.pk 
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/kp_3ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run736374/config1.pk 
/home/energy/amawi/projects/nn_ansatz/src/experiments/HEG/baselines/kp_3ins/kfac_1lr-3_1d-4_1nc-4_m512_el14_s64_p32_l2_det1/run876476/config1.pk)
 
for i in "${!ngpus[@]}"
do
    ngpu=${ngpus[i]}
    path="-p ${paths[i]}"
    echo $path
    sbatch --gres=gpu:RTX3090:$ngpu --job-name=p $submission_path $path
    echo ngpu $ngpu
done

