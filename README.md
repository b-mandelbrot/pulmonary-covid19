## Title

Pulmonary COVID-19: Learning Spatialtemporal Features Combining CNN and LSTM Networks for Lung Ultrasound Video Classification (EN)

## Abstract

Deep Learning is a very active and important area for building Computer-Aided Diagnosis (CAD) applications. This work aims to present a hybrid model to classify lung ultrasound (LUS) videos captured by convex transducers to diagnose COVID-19. A Convolutional Neural Network (CNN) performed the extraction of spatial features, and the temporal dependence was learned using a Long Short-Term Memory (LSTM). Different types of convolutional architectures were used for feature extraction. The hybrid model (CNN-LSTM) hyperparameters were optimized using the Optuna framework. The best hybrid model was composed of an Xception pre-trained on ImageNet and an LSTM containing 512 units, configured with a dropout rate of 0.4, two fully connected layers containing 1024 neurons each, and a sequence of 20 frames in the input layer (20x2018). The model presented an average accuracy of 93% and sensitivity of 97% for COVID-19, outperforming models based purely on spatial approaches. Furthermore, feature extraction using transfer learning with models pre-trained on ImageNet provided comparable results to models pre-trained on 185 videos of LUS. The results corroborate with other studies showing that this model for LUS classification can be an important tool in the fight against COVID-19 and other lung diseases.

## How to reproduce the article?

### Dataset

The lung ultrasound dataset can be accessed at: [covid19_ultrasound](https://github.com/jannisborn/covid19_ultrasound/tree/9e254a140b4faa2c200b8bb5cee2347b7198fbef)

To reproduce the article, it will be necessary to clone the git repository provided above and follow the steps in the [README](https://github.com/jannisborn/covid19_ultrasound/blob/9e254a140b4faa2c200b8bb5cee2347b7198fbef/data/README.md) file.

### Features

All feature sets extracted by the different CNN architectures used in this work are available for download at: [data/features](https://drive.google.com/drive/folders/1dlkpyQ2RrkCi1g8CfZsXYzxqL4X6XFJU?usp=sharing).

### Hybrid model

The hybrid model (Xception-CNN) proposed by this work is available for download in h5 format at: [data/best_model](https://drive.google.com/drive/folders/1dlkpyQ2RrkCi1g8CfZsXYzxqL4X6XFJU?usp=sharing).
