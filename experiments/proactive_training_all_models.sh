#!/bin/bash

full_retraining_prefix="Cifar10/full_retraining"
base_dir="${HOME}/continuous-deployment"
start=0
saved_models=${base_dir}/saved/models
saved_log=${base_dir}/saved/log
best_epoch=50
best_model="checkpoint-epoch${best_epoch}.pth"

exp_prefix="Cifar10/online_training"
echo "Cleaning old logs and models"
rm -rf ${saved_models}/${exp_prefix}
rm -rf ${saved_log}/${exp_prefix}
compress_type="no"
# online training

#for model_name in resnet18 resnet34 resnet50 resnext50_32x4d wide_resnet50_2 densenet161 mobilenet_v2
for model_name in alexnet mnasnet1_0 resnet101 resnet18 resnet50 resnext50_32x4d vgg16 wide_resnet50_2 densenet161 mobilenet_v2 resnet152 resnet34 resnext101_32x8d squeezenet1_1 wide_resnet101_2
do
    for end in 10000 20000 30000 40000
    do
        echo "Running ${model_name} until datapoint #${end}"
        python proactive_train.py --resume ${saved_models}/${full_retraining_prefix}/0_${end}/${model_name}/0/${best_model} \
        --config ${base_dir}/experiments/config-torchvision-proactive.json --model_name ${model_name} \
        --name ${exp_prefix}/${end}_50000/${model_name}/sparse_${compress_type} \
        --history_end ${end} --compress_type no --online_training true --trigger_size 128
    done
done

# normal training
exp_prefix="Cifar10/proactive_training"
echo "Cleaning old logs and models"
rm -rf ${saved_models}/${exp_prefix}
rm -rf ${saved_log}/${exp_prefix}
for model_name in alexnet mnasnet1_0 resnet101 resnet18 resnet50 resnext50_32x4d vgg16 wide_resnet50_2 densenet161 mobilenet_v2 resnet152 resnet34 resnext101_32x8d squeezenet1_1 wide_resnet101_2
do
    for end in 10000 20000 30000 40000
    do
        echo "Running ${model_name} until datapoint #${end}"
        python proactive_train.py --resume ${saved_models}/${full_retraining_prefix}/0_${end}/${model_name}/0/${best_model} \
        --config ${base_dir}/experiments/config-torchvision-proactive.json --model_name ${model_name} \
        --name ${exp_prefix}/${end}_50000/${model_name}/sparse_${compress_type} \
        --history_end ${end} --compress_type no --online_training false --trigger_size 8
    done
done


# sparse training
for model_name in alexnet mnasnet1_0 resnet101 resnet18 resnet50 resnext50_32x4d vgg16 wide_resnet50_2 densenet161 mobilenet_v2 resnet152 resnet34 resnext101_32x8d squeezenet1_1 wide_resnet101_2
do
    for compress_type in randomk topk
    do
        for end in 10000 20000 30000 40000
        do
            for deployment_ratio in 0.01 0.001 0.0001
            do
                echo "Running ${model_name} until datapoint #${end} with compress_type ${compress_type} deployment_ratio ${deployment_ratio}"
                python proactive_train.py --resume ${saved_models}/${full_retraining_prefix}/0_${end}/${model_name}/0/${best_model} \
                --config ${base_dir}/experiments/config-torchvision-proactive.json --model_name ${model_name} \
                --name ${exp_prefix}/${end}_50000/${model_name}/sparse_${compress_type}/d_${deployment_ratio} \
                 --history_end ${end} --compress_type ${compress_type} \
                 --online_training false --deployment_ratio ${deployment_ratio} --trigger_size 8
            done
        done
    done
done
