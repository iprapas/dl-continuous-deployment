#!/bin/bash

for deployment_ratio in 0.001 0.01 0.1
do
    python proactive_train.py --resume saved/models/Cifar_Resnet/0615_202244/checkpoint-epoch50.pth --config config-proactive.json --deployment_ratio ${deployment_ratio} --name proactive_resnet_d${deployment_ratio}
done


