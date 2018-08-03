# Skincancer Project
Original code *by David Soto*  - dasoto@gmail.com

This is a forked version by Freddie Poser  - https://vogonjeltz.com

## 1. Data
The data from this project comes from the [ISIC Archive](https://isic-archive.com/#images) which contains a huge number of images of melanomas which have been used here to train models to classify the risk of a mole turning malignant.

The download feature of the ISIC archive website, whilst powerful, faills frequently resulting in corrupt downloads. Depending on the licencing terms I may upload the selected images used here to a webserver so that people can have easier access to them.

The data used in this project comes from four of the datasets on the ISIC website:
* UDA-1: 60 images `[B: 23, M:37]`
* UDA-2: 557 images `[B: 398, M: 159]`
* MSK-1: 1100 images `[B: 787, M: 301]`
    * I have so far struggled to download this dataset so I have so far excluded it
* MSK-2: 352 images `[B: 0, M: 352]`
    * Benign images here excluded as per orriginal project
    
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

### Problems with keras
On my installation (and it seems others) the use of the keras method `model.fit_generator` leads to a `TypeError` that occurs when trying to pickle an object. For now I have stopped using this and replaced it with a standard `model.fit` call.

## Disclaimer
This tool has been designed only for educational purposes to demonstrate the use of Machine Learning tools in the medical field. This tool does not replace advice or evaluation by a medical professional. Nothing on this site should be construed as an attempt to offer a medical opinion or practice medicine.
