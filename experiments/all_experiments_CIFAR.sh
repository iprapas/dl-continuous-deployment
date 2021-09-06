#!/bin/bash

base_dir="${HOME}/continuous-deployment"
best_epoch=25
experiment="exp1/CIFAR"

for model_name in mobilenet_v2 resnet18 resnet50 densenet161
do
    start=0
    initial_prefix="${experiment}/${model_name}/full_retraining"
    initial_log_dir="${base_dir}/saved/log/${initial_prefix}"
    initial_model_dir="${base_dir}/saved/models/${initial_prefix}"
    echo "Cleaning old logs and models"
    rm -rf ${initial_log_dir}
    rm -rf ${initial_model_dir}
    for end in 10000 20000 30000 40000
    do
        echo "Training on datapoints (${start}-${end})"
        python ${base_dir}/train.py \
        --config ${base_dir}/experiments/config-cifar-initial.json \
        --start ${start} --end ${end} \
        --model_name ${model_name} \
        --epochs ${best_epoch} --save_period ${best_epoch} \
        --name ${initial_prefix}/${start}_${end}
    done

    rm ${base_dir}/tmp/*.json

    for start in 10000 20000 30000 40000
    do
        end=$((start+10000))
        echo "Testing on datapoints (${start}-${end})"
        python ${base_dir}/full_test.py \
        --config experiments/config-cifar-test.json \
        --start ${start} --end ${end} \
        --model_name ${model_name} \
        --resume ${initial_model_dir}/0_${start}/0/checkpoint-epoch${best_epoch}.pth \
        --name ${initial_prefix}

    done

    exp_prefix="${experiment}/${model_name}/online"
    log_dir="${base_dir}/saved/log/${exp_prefix}"
    model_dir="${base_dir}/saved/models/${exp_prefix}"

    echo "Cleaning old logs and models"
    rm -rf ${log_dir}
    rm -rf ${model_dir}

    python ${base_dir}/proactive_train.py \
    --config experiments/config-cifar-proactive.json \
    --resume ${initial_model_dir}/0_10000/0/checkpoint-epoch${best_epoch}.pth \
    --compress_type no \
    --online_training yes \
    --trigger_size 128 \
    --model_name ${model_name} \
    --train_until $(( best_epoch+1 )) \
    --name ${exp_prefix}


    exp_prefix="${experiment}/${model_name}/proactive"
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
            --config experiments/config-cifar-proactive.json \
            --resume ${initial_model_dir}/0_10000/0/checkpoint-epoch${best_epoch}.pth \
            --compress_type ${compress_type} \
            --model_name ${model_name} \
            --train_until $(( best_epoch+1 )) \
            --trigger_size ${trigger_size} \
            --name ${exp_prefix}/sparse_${compress_type}/trigger_${trigger_size}
        done
    done


#    exp_prefix="${experiment}/${model_name}/proactive"
#    log_dir="${base_dir}/saved/log/${exp_prefix}"
#    model_dir="${base_dir}/saved/models/${exp_prefix}"

#    echo "Cleaning old logs and models"
#    rm -rf ${log_dir}
#    rm -rf ${model_dir}

    for compress_type in topk
    do
        for d in 0.01 0.001 0.0001
        do
            python ${base_dir}/proactive_train.py \
            --config experiments/config-cifar-proactive.json \
            --resume ${initial_model_dir}/0_10000/0/checkpoint-epoch${best_epoch}.pth \
            --deployment_ratio ${d} \
            --compress_type ${compress_type} \
            --model_name ${model_name} \
            --train_until $(( best_epoch+1 )) \
            --name ${exp_prefix}/sparse_${compress_type}/d_${d}
        done
    done
done