# Analyzing the Generalization Capability of SGLD Using Properties of Gaussian Channels

This repository is the official implementation of Neurips 2021 paper "Analyzing the Generalization Capability of SGLD Using Properties of Gaussian Channels", written in Python3.

In this paper, we investigate the generalization gap of the stochastic gradient Langevin dynamics (SGLD) algorithm.  We derive a generalization bound by using properties of Gaussian channels, which have been well understood in information and communication theory. Our bound incorporates the variance of gradients for quantifying a particular kind of ¡°sharpness¡± of the loss landscape. We also consider a closely related algorithm with SGLD, namely differentially private SGD (DP-SGD). We prove that the generalization capability of DP-SGD can be amplified by iteration. Specifically, our bound can be sharpened by including a time-decaying factor if the DP-SGD outputs the last iterate while keeping other iterates hidden. This decay factor enables the contribution of early iterations to our bound to reduce with time and is established by strong data processing inequalities¡ªa fundamental tool in information theory. We demonstrate our bound through numerical experiments, showing that it can predict the behavior of the true generalization gap.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

This command will install the packages needed for executing the code. The datasets needed for reproducing the results will be automatically downloaded during the inital execution of the code.

Warning: This program uses a lot of storage space! The storage usage grows linearly with number of trainable parameters in the model, as well as the number of iterations during training. It is recommended to have at least 20GB of available space for reproducing the results in the paper.

## Execution

To execute the program we first create a directory "gradient" in same directory you put the "main.py" file. Then create a "models" directory and put the code defining the models in it. The default "fc.py" and "cnn.py" for reproducing the results are already provided. Then run one single command:

```
python3 main.py --dataset [DATASET] --model [MODEL] --epochs [NUMBER OF TRAINING EPOCHS] --batchsize [BATCHSIZE] --learningrate [LEARNING RATE] --label_corrupt_prob [FRACTION OF DATA THAT IS REPLACED WITH RANDOM LABELS] --num_sample_path [NUMBER OF SAMPLES PATHS IN A SINGLE EXPERIMENT]
```

The parameters for each of the experiments are all provided in the paper. You can also try other parameters, but be aware of storage usage as explained previously.

## Results

The "data" directory contains raw outputs from the experiments we have run, which are used to create the plots and tables in the paper. You can use these data to reproducing the plots without re-running the code. You can also use them as a sanity check against the outputs of your own runs of the program.

## Acknowledgement

The code is partially based on the code written by Behnam Neyshabur, et. al. for NIPS 2017 paper "Exploring Generalization in Deep Learning" (https://arxiv.org/abs/1706.08947). We thank them for generously permitting us for reusing the code.

## License

The experiments in the paper run on two datasets: MNIST and CIFAR-10, under the license of Creative Commons Attribution-Share Alike 3.0 and MIT license, respectively. 