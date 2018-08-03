from keras.preprocessing.image import ImageDataGenerator
from ml_lib.cnn_model import CNN
from ml_lib.moleimages import MoleImages
from ml_lib.roc import plot_roc

import tensorflow as tf
from sklearn.metrics import classification_report

train_data_dir = 'data_scaled/'
validation_data_dir = 'data_scaled_validation/'
nb_train_samples = 873
nb_validation_samples = 96
epochs = 2
batch_size = 10

#model = mycnn.fit_generator(train_generator,validation_generator,
#    nb_train_samples, nb_validation_samples, epochs, batch_size)

mimg = MoleImages()

X_test, y_test = mimg.load_test_images('data_scaled_validation/benign', 'data_scaled_validation/malign')

X_train, y_train = mimg.load_test_images('data_scaled/benign', 'data_scaled/malign')

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

mycnn = CNN()

for i in range(0, 100):
    mycnn.fit(
        X_train, y_train,
        X_test, y_test,
        1, batch_size
    )

    mycnn.save_model("models/model_1_test.h5")

    y_pred_proba = mycnn.predict(X_test)
    y_pred = (y_pred_proba > 0.5) * 1
    print(classification_report(y_test, y_pred))
    plot_roc(y_test, y_pred_proba, title='ROC Curve CNN from scratch ' + str(i))