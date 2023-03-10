# CALPAGAN: CALorimetry for PArticles using Generative Adversarial Networks

This repository contains one of the experiments carried out for the work "CALPAGAN" (presented in the conference [YEFIST](http://www.yefist.org/), on  September 25, 2022).

As a model, we use Pix2pix, a conditional generative adversarial networks (cGAN), proposed in the article [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004).

---

## Requirements

This repository is prepared to train the model entirely on Google Colab. That means, to train the model and generate the images without any issue, it is sufficient to

- open the notebook [`calpagan_train_test.ipynb`](https://github.com/byrkbrk/calpagan-experiment/blob/0a29eaa9617c3e1f7290a8d61cdd414eac54eda0/calpagan_train_test.ipynb) 
on Google Colab (using the [link](https://colab.research.google.com/github/byrkbrk/calpagan-experiment/blob/main/calpagan_train_test.ipynb) or the link *Open in Colab* inside the notebook)
- log in your Google account (if you haven't yet)
- run the (self-explanatory) notebook cells one by one

## Training dataset

Our training dataset consists of 5000 Delphes and Geant4 (simulation) images (which represent condition and real images, respectively). One examplar pair from the dataset and the corresponding jet reconstructions are presented below:

![delphes-geant4-jets](./images-for-readme/delphes-geant4-jets.png)

## Test dataset and image generation

Similary, our test dataset contains 4999 Delphes and Geant4 image pairs that serve as condition and ground truth examples, respectively. One examplar of the generated images (i.e., fake) for the provided condition (i.e., Delphes) and real (i.e., Geant4) images is as follows:

![condition-real-fake](./images-for-readme/condition-real-fake.png)
