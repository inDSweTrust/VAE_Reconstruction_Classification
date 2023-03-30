# VAE_Reconstruction_Classification

This repository contains a Pytorch implementation of a VAE for CIFAR10 reconstruction, generation and classification. This work is a part of a research project on VAE latent space dimension optimization.

The goal is to minimize the VAE latent space while keeping high reconstruction capacity and classification accuracy.

![vae_recon](https://user-images.githubusercontent.com/68122114/228701219-738d4f69-161b-48b7-99a1-83d85b4e389d.png)


## Pipeline

The implemented pipeline is shown below. CIFAR10 images are encoded with a VAE, where the dimension of the encoding (latent_dim) is the experimental variable. The encoding is then used for the following tasks:
- image reconstruction 
- linear classification 

![Screenshot 2023-03-29 175735](https://user-images.githubusercontent.com/68122114/228700782-88b47161-2f9e-4d34-995a-529c03d678f6.png)

## Install and run project

### VAE Reconstruction

```python
./train.py -rd path/to/data -dim LATENT_DIM
```
### Classifier

```python
./train.py -p path/to/encoded_data -d LATENT_DIM
```

## References
[1] “Examining the Size of the Latent Space of Convolutional Variational Autoencoders Trained With Spectral Topographic Maps of EEG Frequency Bands”, TAUFIQUE AHMED, LUCA LONGO

