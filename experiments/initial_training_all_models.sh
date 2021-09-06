#!/bin/bash

exp_prefix="Cifar10/full_retraining"
base_dir="${HOME}/continuous-deployment"
start=0

echo "Cleaning old logs and models"
rm -rf ${base_dir}/saved/log/${exp_prefix}
rm -rf ${base_dir}/saved/models/${exp_prefix}

for model_name in resnet18 resnet34 resnet50 resnet101 resnet152 resnext50_32x4d resnext101_32x8d wide_resnet50_2 wide_resnet101_2 mnasnet1_0 mobilenet_v2 alexnet vgg16 squeezenet1_1 densenet161
do
    for end in 10000 20000 30000 40000 50000
    do
        echo "Running ${model_name} until datapoint #${end}"
        time python train.py --config ${base_dir}/experiments/config-torchvision-initial.json --model_name ${model_name} --name ${exp_prefix}/${start}_${end}/${model_name} --start ${start} --end ${end}
    done
done
