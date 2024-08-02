
---

# Enhancing Glassy Dynamics Prediction by Incorporating Displacement from Initial to Equilibrium States

This repository contains the official implementation of the EIGNN model.


## Raw Dataset

The EIGNN model is trained and tested using the [GlassBench](https://doi.org/10.5281/zenodo.10118191) dataset, created by G. Jung and presented in the [Roadmap on Machine Learning for Glassy Liquids](https://arxiv.org/abs/2311.14752). This dataset includes initial positions, inherent positions, and cage positions associated with each configuration. The initial positions are sourced from Shiba's [dataset](https://ipomoea-www.cc.u-tokyo.ac.jp/i29002/botan/public_dataset.tar.gz), which was introduced in the paper [BOTAN: BOnd TArgeting Network for Prediction of Slow Glassy Dynamics by Machine Learning Relative Motion](https://pubs.aip.org/aip/jcp/article/158/8/084503/2868947/BOTAN-BOnd-TArgeting-Network-for-prediction-of).

## Reproduce the Results

To facilitate the reproduction of the results of the EIGNN-cage model, we provide a [Google Colab notebook](https://github.com/xjiang-hnu/EIGNN/blob/main/Training%20and%20Test%20with%20EIGNN_Cage.ipynb). This notebook documents the entire process in detail, including data preprocessing, model initialization, and the training and testing procedures at a temperature of T=0.44. Here’s a summary of the Colab:

1. **Data Preprocessing**: The notebook describes the steps for cleaning, transforming, and preparing the raw data for model training.
2. **Model Initialization**: It includes a detailed description of the model architecture, along with the hyperparameters and configurations used.
3. **Training and Testing**: The notebook outlines the procedures for training the model at T=0.44, including the training loop, loss function, and evaluation metrics.
### Traing and Test Reuslts
In this Colab notebook, the trainging and testing results are recorded in the output cells. You can check that the finnl test reuslts of the EIGNN-cage model are as follows:


| T = 0.44 |  0.13   | 1.30     | 13.0    | 130     | 412    | 1300      | 4120     | 13 000      | 41 200      | 130 000    |
|----------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|--------------------|
| Pcc(mean)           | 0.87364581        | 0.87064157        | 0.91661659        | 0.86220094        | 0.84399332        | 0.8393544         | 0.82280353        | 0.75662464        | 0.58931166        | 0.30577556         |
| Pcc(all)          | 0.87378081        | 0.87046921        | 0.91590845        | 0.86153974        | 0.84486262        | 0.8401456         | 0.8226806         | 0.74941987        | 0.55762555        | 0.29931875         |
### Steps to Reproduce:

1. **Downloaded the Raw Dataset**: Download the original dataset from GlassBench.  
2. **Set the Path**: Open the provided Colab notebook and ensure all necessary dependencies are installed. Set the root path for the Raw dataset. Set the code path for the [model](https://github.com/xjiang-hnu/EIGNN/tree/main/model).
3. **Load the Model Checkpoint**: Navigate to the section where the model checkpoint is loaded.
4. **Run the Code**: Execute the provided code to load the provied trained model and run the testing code to verify the results.

## Results in the paper 

The results for Fig.3, Fi.4 and Fig.5 can be found in [Results floader](https://github.com/xjiang-hnu/EIGNN/tree/main/results).
