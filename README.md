Sure, here's an expanded and refined version of the text:

---

# Enhancing Glassy Dynamics Prediction by Incorporating Displacement from Initial to Equilibrium States

This repository contains the official implementation of the EIGNN model.

## Raw Dataset

The EIGNN model is trained and tested using the [GlassBench](https://doi.org/10.5281/zenodo.10118191) dataset, curated by G. Jung and presented in the [Roadmap on Machine Learning for Glassy Liquids](https://arxiv.org/abs/2311.14752). This dataset includes initial positions, inherent positions, and cage positions associated with each configuration. The initial positions are sourced from Shiba's [dataset](https://ipomoea-www.cc.u-tokyo.ac.jp/i29002/botan/public_dataset.tar.gz), which was introduced in the paper [BOTAN: BOnd TArgeting Network for Prediction of Slow Glassy Dynamics by Machine Learning Relative Motion](https://pubs.aip.org/aip/jcp/article/158/8/084503/2868947/BOTAN-BOnd-TArgeting-Network-for-prediction-of).

## Reproduce the Results

To facilitate the reproduction of the results of the EIGNN-cage model, we provide a Colab notebook. This notebook documents the entire process in detail, including data preprocessing, model initialization, and the training and testing procedures at a temperature of t=0.44. Here’s a summary of the process:

1. **Data Preprocessing**: The notebook describes the steps for cleaning, transforming, and preparing the raw data for model training.
2. **Model Initialization**: It includes a detailed description of the model architecture, along with the hyperparameters and configurations used.
3. **Training and Testing**: The notebook outlines the procedures for training the model at t=0.44, including the training loop, loss function, and evaluation metrics.

### Steps to Reproduce:

1. **Downloaded the Raw Dataset**: Download the original dataset from GlassBench.  
2. **Set the Path**: Open the provided Colab notebook and ensure all necessary dependencies are installed. Set the root path for the Raw dataset. Set the model path for the code.
3. **Load the Model Checkpoint**: Navigate to the section where the model checkpoint is loaded.
4. **Run the Code**: Execute the provided code to load the provied trained model and run the testing code to verify the results.

By following these steps, you can easily reproduce the results and validate the outcomes of the study. This ensures transparency and reproducibility, allowing others to build upon this research.

---

Let me know if this captures the details accurately or if you need further modifications.
