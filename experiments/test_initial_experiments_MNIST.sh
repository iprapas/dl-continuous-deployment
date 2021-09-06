#!/bin/bash

exp_prefix="MNIST/full_retraining"
base_dir="${HOME}/Projects/dima-thesis/continuous-deployment"
start=0

log_dir="${base_dir}/saved/log/${exp_prefix}"
model_dir="${base_dir}/saved/models/${exp_prefix}"

echo "Cleaning old logs and models"
rm -rf ${log_dir}
rm -rf ${model_dir}

for end in 10000 20000 30000 40000 50000
do
    echo "Running ${model_name} until datapoint #${end}"
    python ${base_dir}/train.py --config ${base_dir}/experiments/config-lenet-initial.json --name ${exp_prefix}/${start}_${end}/${model_name} --start ${start} --end ${end}
done

#for start in 10000 20000 30000 40000 50000
#do
#    python ${base_dir}/full_test.py --resume ${model_dir}/0_${start} --config ${base_dir}/experiments/config-lenet-initial.json --start ${start} --end ${end}
#done
