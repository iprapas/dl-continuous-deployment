#!/bin/bash

base_dir="${HOME}/continuous-deployment"
start=0

experiment="exp1/MNIST"
initial_prefix="${experiment}/LeNet/full_retraining"
initial_log_dir="${base_dir}/saved/log/${initial_prefix}"
initial_model_dir="${base_dir}/saved/models/${initial_prefix}"

echo "Cleaning old logs and models"
rm -rf ${initial_log_dir}
rm -rf ${initial_model_dir}

for end in 10000 20000 30000 40000 50000
do
    echo "Training on datapoints (${start}-${end})"
    python ${base_dir}/train.py --config ${base_dir}/experiments/config-lenet-initial.json --name ${initial_prefix}/${start}_${end}/${model_name} --start ${start} --end ${end}
done

rm ${base_dir}/tmp/*.json


for start in 10000 20000 30000 40000 50000
do
    end=$((start+10000))
    echo "Testing on datapoints (${start}-${end})"
    python ${base_dir}/full_test.py --config experiments/config-lenet-test.json \
    --start ${start} --end ${end} \
    --resume ${initial_model_dir}/0_${start}/0/checkpoint-epoch10.pth \
    --name ${initial_prefix}
done


exp_prefix="${experiment}/LeNet/online"
log_dir="${base_dir}/saved/log/${exp_prefix}"
model_dir="${base_dir}/saved/models/${exp_prefix}"

echo "Cleaning old logs and models"
rm -rf ${log_dir}
rm -rf ${model_dir}


python ${base_dir}/proactive_train.py \
--config experiments/config-lenet-proactive.json \
--resume ${initial_model_dir}/0_10000/0/checkpoint-epoch10.pth \
--compress_type no \
--online_training yes \
--trigger_size 128 \
--name ${exp_prefix}


exp_prefix="${experiment}/LeNet/proactive"
log_dir="${base_dir}/saved/log/${exp_prefix}"
model_dir="${base_dir}/saved/models/${exp_prefix}"

echo "Cleaning old logs and models"
rm -rf ${log_dir}
rm -rf ${model_dir}

for trigger_size in 8 16 32 64
do
    for compress_type in no
    do
        python ${base_dir}/proactive_train.py \
        --config experiments/config-lenet-proactive.json \
        --resume ${initial_model_dir}/0_10000/0/checkpoint-epoch10.pth \
        --compress_type ${compress_type} \
        --trigger_size ${trigger_size} \
        --name ${exp_prefix}/sparse_${compress_type}/trigger_${trigger_size}
    done
done


#exp_prefix="MNIST/LeNet/proactive"
#log_dir="${base_dir}/saved/log/${exp_prefix}"
#model_dir="${base_dir}/saved/models/${exp_prefix}"
#
#echo "Cleaning old logs and models"
#rm -rf ${log_dir}
#rm -rf ${model_dir}

for compress_type in randomk topk
do
    for d in 0.01 0.001 0.0001
    do
    python ${base_dir}/proactive_train.py \
    --config experiments/config-lenet-proactive.json \
    --resume ${initial_model_dir}/0_10000/0/checkpoint-epoch10.pth \
    --deployment_ratio ${d} \
    --compress_type ${compress_type} \
    --name ${exp_prefix}/sparse_${compress_type}/d_${d}
    done
done

