# Official Repository for Atari Implementation of GABRIL

In this repository, we provide the code and instructions to run **Atari** experiments from our paper: [**GABRIL: Gaze-Based Regularization for Mitigating Causal Confusion in Imitation Learning**](https://liralab.usc.edu/gabril/).

<p align="center">
  <img src="images/method.png" alt="" width="500"/>
</p>


## Requirements

This repo is heavily based on the implementation of OREO for behavioral cloning in Atari environments. To begin with, clone our repository and install the required packages:

```
git clone https://github.com/nfbahrani/GABRIL-Atari.git
cd GABRIL-Atari

pip install -r requirements.txt
```

## Dataset 

We used the Atari library provided by gymnasium. Specifically, we used ```frameskip=4``` and ```sticky_action``` probability ```0.25``` for a bunch of 15 selected Atari games. The recordings contain 84x84 grayscale images of the game while playing, in addition to the collected gaze coordinates, and the discrete actions. 

Please download the dataset.zip file [here](https://drive.google.com/drive/folders/1hhMzlrbMKK8dcv526pxthsjJuEtuh-bJ) and extract the files to see the following structure:

```bash
GABRIL-Atari
├── dataset
    ├── Alien
    ├── Assault
    ├── Asterix
    ...
├── train_bc.py             # main code for training
├── utils.py                # main code for loading the dataset and evaluating trained models
```

## Pre-Training Gaze Predictor

Some of the gaze-based methods implemented in this repository requires access to a pretrained gaze mask predictor. For example, ViSaRL needs access to an external gaze predictor both during train and test. One can train such gaze predictor as

```
python train_gaze_predictor.py
```

Note that the gaze predictor here is different from the gaze predictor we introduced in our paper. In fact, the gaze predictor here can be seen as a pretraining step, consuming all gaze samples before the main BC training. The pretrained gaze predictor is then frozen and serves as the source of gaze mask generation later during the main traning.

## Training

To train a *regular BC* agent in the normal environment, run

```
python train_bc.py --gaze_method=None
```

To train *GABRIL* in normal environment, run

```
python train_bc.py
```

and for the confounded environement, run

```
python train_bc.py --train_type=confounded --eval_type=confounded
```

To train *GABRIL+GMD*, run

```
python train_bc.py --dp_method=GMD
```

Model evaluation happens every 100 epochs of training and the result is printed to the console.

## References

* [OREO](https://github.com/alinlab/oreo)
