from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


from ml_lib.cnn_model import CNN
from ml_lib.moleimages import MoleImages
from ml_lib.roc import plot_roc

import tensorflow as tf
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from keras.backend.tensorflow_backend import set_session

import sys

config = tf.ConfigProto()
#config.log_device_placement=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
set_session(sess)


train_data_dir = 'data_scaled/'
validation_data_dir = 'data_scaled_validation/'
nb_train_samples = 1853
nb_validation_samples = 204
batch_size = 32

#model = mycnn.fit_generator(train_generator,validation_generator,
#    nb_train_samples, nb_validation_samples, epochs, batch_size)

mimg = MoleImages()

X_test, y_test = mimg.load_test_images('data_scaled_validation/benign', 'data_scaled_validation/malign')

X_train, y_train = mimg.load_test_images('data_scaled/benign', 'data_scaled/malign')

mycnn = CNN()

areas = []
accuracies = []

for i in range(0, 30):
    score = mycnn.fit(
        X_train, y_train,
        X_test, y_test,
        5, batch_size
    )

    accuracies.append(score[1])

    mycnn.save_model("models/model_1_test.h5")

    y_pred_proba = mycnn.predict(X_test)
    y_pred = (y_pred_proba > 0.5) * 1
    print(classification_report(y_test, y_pred))
    area = plot_roc(y_test, y_pred_proba, title='ROC Curve CNN from scratch Epoch:' + str((i + 1) * 5))
    print("Area_ROC = {}".format(area))
    areas.append(area)

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(accuracies, color = color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("AUC")
    ax2.plot(areas, color = color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Accuracy and AUC over time")
    plt.show()