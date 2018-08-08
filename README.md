# Skincancer Project
Original code *by David Soto*  - dasoto@gmail.com

Original repo can be found [here](https://github.com/dasoto/skincancer)

This is a forked version by Freddie Poser  - https://vogonjeltz.com

## 1. Data
The data from this project comes from the [ISIC Archive](https://isic-archive.com/#images) which contains a huge number of images of melanomas which have been used here to train models to classify the risk of a mole turning malignant.

The download feature of the ISIC archive website, whilst powerful, faills frequently resulting in corrupt downloads. Depending on the licencing terms I may upload the selected images used here to a webserver so that people can have easier access to them.

The data used in this project comes from four of the datasets on the ISIC website:
* UDA-1: 60 images `[B: 23, M:37]`
* UDA-2: 557 images `[B: 398, M: 159]`
* MSK-1: 1100 images `[B: 787, M: 301]`
* MSK-2: 352 images `[B: 0, M: 352]`
    * Benign images here excluded as per original project
    
### Data Processing and Ingest
To set the data up for the project to use methods can be found in the `src/image_preparation.py` and `src/moleimages.py` class.

To prepare:

1) Set up the following directory structure:

    ```
    \-skincancer
        \-data
            \-benign
            |    \- All benign images as JPG...
            |-malignant
                 \- All malignant images as JPG...
        \data_scaled
            \-benign
            |-malgnant
        \data_scaled_validation
            \-benign
            |-malignant
    ```
2) Run the methods in `src/image_preparation.py`. First `resize_images()` then `cv_images`

The `data_scaled` directory will now contain the scaled data for use in training and `data_scaled_validation` contains images to be used for validation (by default 10% of the provided images).

## 2. Training 
### Attempt One
Once the issue in keras (described below) was avoided I managed to start getting the CNN used in the original project to train on the data set. This resuslted (after 95 epochs) and the ROC curve below.

![ROC Curve AUC=0.85][roc_v1_95_85]

My next problem was to get GPU training working on my machine. This was more of a hassle than it should have been but in the end I got it working with the following versions:

* `tensorflow-gpu==1.9.0`
* `CUDA 9.0`
* `cudNN 7.0.5`
* Python 3.6
* Latest version of drivers for my card (960M)

This was done in  an Anaconda3 environment. Installed using the following commands (once CUDA/cuDNN installed):
* `conda install numpy`
* `pip install tensorflow-gpu`
* `pip install keras`

This installation sped up training by ~8x.

The code for all of this can be found in `src/sandbox/v1_playground.py`

### Notebook Visualisations

Included in the original project was a jupyter notebook that was used to visualise the output of a network. I updated this file to work better with multiple images.

### VGG16 and Data Augmentation
As in the original project I have attempted to use transfer learning and apply the VGG16 model (pre-trained on the ImageNet dataset) to the problem. The code can be found in the two files `src/sandbox/v2_vgg.py` and `src/sandbox/v3_vgg_DA.py` as well as the file containing the model definition `src/sandbox/ml_lib/vgg_model.py`.

The structure I used was to take the VGG model (without the last three layers) and feed it into a fully connected layer with 128 neurons. This then has one output - for binary classification. i freeze the whole of the VGG16 network so that the weights don't change during training. This leaves just the final two layers that learn to interpret the result of the VGG output.

As well as learning better than the standalone CNN it also learns a lot faster, skipping out the stage I observed where the network would not train at all for up to 30 epochs before changing. This network reached ~0.73 accuracy in just three epochs.

I also used data-augmentation (rotation and flipping) techniques as well as dropout to prevent overfitting.

I trained this structure for 90 epochs which gave the following results:

![ROC Curve for the VGG Network trained for 90 epochs][roc_v3_90e]
![Accuracy and AUC over training][acc_auc_v3_90e]

| class | precision | recall  | f1-score  | support |
|:----- |:-----     | :--     |:--        |:--      |
| 0.0   | 0.79     | 0.96    |  0.86     |    120   |
| 1.0   | 0.91      | 0.63    | 0.75      |   84    |
|avg / total|0.84   | 0.82    |0.82       |204      |

### "Combined" Model

I had the idea to try and combine the VGG16 model with a small standard CNN (3 layers) to try and achieve better results. I am still experimenting with this setup but my initial test trained for 150 epochs achieved an AUC of 90 and test results as follows:

| class | precision | recall  | f1-score  | support |
|:----- |:-----     | :--     |:--        |:--      |
| 0.0   | 0.84     | 0.90    |  0.87     |    120   |
| 1.0   | 0.84      | 0.75    | 0.79      |   84    |
|avg / total|0.84   | 0.84    |0.84       |204      |


![ROC for the Combined Model trained for 90 epochs][roc_combined_150e]

## 3. Other

### Problems with keras
On my installation (and it seems others) the use of the keras method `model.fit_generator` leads to a `TypeError` that occurs when trying to pickle an object. For now I have stopped using this and replaced it with a standard `model.fit` call. This persists when using `tensorflow-gpu 1.9.0`

When trying to implement DA using the inbuilt keras tools I found that the key was to set the option `use_multiprocessing` to `False`. This allows me to use the full features of `fit_generator` without the breaking bug.

## Disclaimer
This tool has been designed only for educational purposes to demonstrate the use of Machine Learning tools in the medical field. This tool does not replace advice or evaluation by a medical professional. Nothing on this site should be construed as an attempt to offer a medical opinion or practice medicine.

[roc_v1_95_85]:https://github.com/vogon101/skincancer/blob/master/results/Initial%20Testing/ROC%20Curve%20-%2085.png
[roc_v3_90e]:https://github.com/vogon101/skincancer/blob/master/results/Transfer%20Learning%20with%20DA/1-roc.png
[acc_auc_v3_90e]:https://github.com/vogon101/skincancer/blob/master/results/Transfer%20Learning%20with%20DA/1-acc-auc.png
[roc_combined_150e]:https://github.com/vogon101/skincancer/blob/master/results/Combined%20Model/1-150epoch-roc.png
[arch_combined_1]:https://github.com/vogon101/skincancer/blob/master/results/Combined%20Model/1-model.png
[acc_auc_combined_150e]:https://github.com/vogon101/skincancer/blob/master/results/Combined%20Model/1-150epoch-acc-auc.png