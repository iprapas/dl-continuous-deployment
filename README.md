# Continuous Training and Deployment of DL models

This code is based on the thesis I did for the completion of my studies in the Big Data Management and Analytics (BDMA) Erasmus Mundus MSc. 

It was presented in [LWDA 2021](https://mcml.ai/lwda2021/fgdb/) and published in the [Datenbank Spektrum](https://link.springer.com/article/10.1007/s13222-021-00386-8) journal.

In this project, we experiment with continuous training and deployment of Deep Learning models. 

This work draws on previous work on [continuous training](https://github.com/d-behi/continuous-pipeline-deployment) of ML models, which proposes proactive stochastic training with mini-batch SGD using a combination of historical data and new data. 

It is a middleground solution that lies between two extremes:

 1. **Online learning**  updates a model by only training on new samples that arrive in the system.
 2. **Full retraining** trains a new model from scratch using all available data samples, when enough new data are available.
 
 
**Proactive training** reuses trained model and historical data, while incorporating new data as soon as they arrive.

In the code, it is simulated by:

`data_loader/data_loaders.py : BatchRandomDynamicSampler`

Training as soon as data arrives, requires a way to quickly deploy the model. For this, we propose sparse continuous deployment.

Drawing from work on the distributed training setting, we propose gradient sparsification. 
This means only changing a small percentage of the model parameters at each iteration
keeping residuals in gradient memory for the changes that did not happen.

The logic of the sparse training and deployment is in `trainer/sparse_trainer.py`

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements

* This project template is created using the [Pytorch-Project-Template](https://github.com/SunQpark/pytorch-template)
* get_top_k, get_random_k gradient sparsification functions are adjusted from [ChocoSGD](https://github.com/epfml/ChocoSGD)
