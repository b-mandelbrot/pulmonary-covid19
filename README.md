## Title

[Pulmonary COVID-19: Learning Spatiotemporal Features Combining CNN and LSTM Networks for Lung Ultrasound Video Classification](https://www.mdpi.com/1424-8220/21/16/5486)

## Journal

[Sensors](https://www.mdpi.com/journal/sensors)

## Section

[Sensing and Imaging](https://www.mdpi.com/journal/sensors/sections/sensing_imaging)

## Special Issue

[Image Sensing and Processing with Convolutional Neural Networks](https://www.mdpi.com/journal/sensors/special_issues/Image_Sensing_Processing)

## Abstract

Deep Learning is a very active and important area for building Computer-Aided Diagnosis (CAD) applications. This work aims to present a hybrid model to classify lung ultrasound (LUS) videos captured by convex transducers to diagnose COVID-19. A Convolutional Neural Network (CNN) performed the extraction of spatial features, and the temporal dependence was learned using a Long Short-Term Memory (LSTM). Different types of convolutional architectures were used for feature extraction. The hybrid model (CNN-LSTM) hyperparameters were optimized using the Optuna framework. The best hybrid model was composed of an Xception pre-trained on ImageNet and an LSTM containing 512 units, configured with a dropout rate of 0.4, two fully connected layers containing 1024 neurons each, and a sequence of 20 frames in the input layer (20x2018). The model presented an average accuracy of 93% and sensitivity of 97% for COVID-19, outperforming models based purely on spatial approaches. Furthermore, feature extraction using transfer learning with models pre-trained on ImageNet provided comparable results to models pre-trained on LUS images. The results corroborate with other studies showing that this model for LUS classification can be an important tool in the fight against COVID-19 and other lung diseases.

## Keywords

COVID-19; CNN; Deep Learning; LSTM; Lung Ultrasound; Neural Networks; Hyperparameter Optimization

## How to replicate the article?

### Dataset

The lung ultrasound dataset can be accessed at: [covid19_ultrasound](https://github.com/jannisborn/covid19_ultrasound/tree/9e254a140b4faa2c200b8bb5cee2347b7198fbef)

To replicate the article will be necessary to clone the git repository provided above and follow the steps in the [README](https://github.com/jannisborn/covid19_ultrasound/blob/9e254a140b4faa2c200b8bb5cee2347b7198fbef/data/README.md) file.

### Features

All feature sets extracted by the different CNN architectures used in this work are available for download at: [data/features](https://drive.google.com/drive/folders/1dlkpyQ2RrkCi1g8CfZsXYzxqL4X6XFJU?usp=sharing).

### Hybrid model

The hybrid model (Xception-LSTM) proposed by this work is available for download in h5 format at: [data/best_model](https://drive.google.com/drive/folders/1dlkpyQ2RrkCi1g8CfZsXYzxqL4X6XFJU?usp=sharing).

### Hyperparameters optimization (HPO)

The database containing the optimization results of all hybrid models is available for download at: [data/optuna](https://drive.google.com/drive/folders/1dlkpyQ2RrkCi1g8CfZsXYzxqL4X6XFJU?usp=sharing)

To see the results, you need to install the `optuna-dashboard` library for `Python 3`.

    $ pip install optuna-dashboard
    $ optuna-dashboard sqlite:///optuna.db # you must have previously downloaded the file.
    
Visit the url [http://127.0.0.1:8080/](http://127.0.0.1:8080/) to view the dashboard.

### Jupyter notebook

To use the notebook is necessary to install some dependencies for `Python 3`.

    $ pip install jupyter numpy sklearn tensorflow==2.4.1

To run the model and extract the metrics provided in the paper use the notebook: [xception-lstm.ipynb](xception-lstm.ipynb).

### Evaluation

The [evaluation.csv](evaluation.csv) file contains the numerical results for each model.

### Cite

	MDPI and ACS Style
	Barros, B.; Lacerda, P.; Albuquerque, C.; Conci, A. Pulmonary COVID-19: Learning Spatiotemporal Features Combining CNN and LSTM Networks for Lung Ultrasound Video Classification. Sensors 2021, 21, 5486. https://doi.org/10.3390/s21165486

	AMA Style
	Barros B, Lacerda P, Albuquerque C, Conci A. Pulmonary COVID-19: Learning Spatiotemporal Features Combining CNN and LSTM Networks for Lung Ultrasound Video Classification. Sensors. 2021; 21(16):5486. https://doi.org/10.3390/s21165486

	Chicago/Turabian Style
	Barros, Bruno, Paulo Lacerda, C??lio Albuquerque, and Aura Conci. 2021. "Pulmonary COVID-19: Learning Spatiotemporal Features Combining CNN and LSTM Networks for Lung Ultrasound Video Classification" Sensors 21, no. 16: 5486. https://doi.org/10.3390/s21165486

### BibTeX

	@article{Barros2021,
	  author = {Barros, Bruno and Lacerda, Paulo and Albuquerque, C{\'{e}}lio and Conci, Aura},
	  doi = {10.3390/S21165486},
	  month = {aug},
	  number = {16},
	  pages = {5486},
	  publisher = {Multidisciplinary Digital Publishing Institute},
	  title = {{Pulmonary COVID-19: Learning Spatiotemporal Features Combining CNN and LSTM Networks for Lung Ultrasound Video Classification}},
	  url = {https://www.mdpi.com/1424-8220/21/16/5486},
	  volume = {21},
	  year = {2021}
	}
