# **Stable Diffusion Image Classification Using PatchGAN and CNN**

## **Introduction**

This project attempts to tackle the inreasingly relevant problem of classifying real images from those generated through sophisticated techniques like stable diffusion. This repository contains two machine learning models that are designed for this problem:
-**PatchGAN Discriminator:** Inspired by the discriminator in a pix2pix generative adversarial network, the PatchGAN discriminator was designed to evaluate whether an image is real or fake by determining whether individual patches of the image are real or fake. This feature will hopefully allow it to learn finer details of the images and better distinguish real images from those generated through stable diffusion.
-**Convolutional Neural Network:** A simple CNN was used as a baseline for comparison against the PatchGAN model. This model was inspired by Sahil Danayak on Kaggle, cited below.
This project uses real images from the CIFAR-10 dataset, and stable diffusion images from the CIFAKE dataset created by Bird & Lofti, both cited below.

## **Technologies Required**
- Python 3.x
- Tensorflow 2.x
- NumPy

## **Installation**
<pre>
git clone https://github.com/nabraun03/stable-diffusion-image-classification.git
cd stable-diffusion-image-classification
pip install -r requirements.txt
</pre>
**Note:** This repository does not contain the dataset used. The CIFAKE dataset can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

## **Usage** 
To run the project after cloning the repository and downloading the dataset, execute the following command:
<pre>
  python main.py
</pre>

