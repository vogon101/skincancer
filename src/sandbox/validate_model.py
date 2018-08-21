import sys

from keras.models import load_model

from ml_lib.moleimages import MoleImages

from ml_lib.roc import classification_report, plot_roc

import tensorflow as tf

from keras.backend import set_session

gpu = 0
if (len(sys.argv) > 2):
    if sys.argv[2] == "-gpu": gpu = 1

config = tf.ConfigProto(
    device_count = {"GPU": gpu}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

model_path = sys.argv[1]

model = load_model(model_path)

mimg = MoleImages()
X_test, y_test = mimg.load_test_images('data_scaled_validation/benign', 'data_scaled_validation/malign')

X_train, y_train = mimg.load_test_images('data_scaled/benign', 'data_scaled/malign')

print("Test Dataset")
y_pred_proba = model.predict(X_test, verbose=1)
y_pred = (y_pred_proba > 0.5) * 1
print(classification_report(y_test, y_pred))
area = plot_roc(y_test, y_pred_proba,
                title="")
print("Area Under ROC = {}".format(area))

print("Train Dataset")
y_pred_proba = model.predict(X_train, verbose=1)
y_pred = (y_pred_proba > 0.5) * 1
print(classification_report(y_train, y_pred))
area = plot_roc(y_train, y_pred_proba,
                title="")
print("Area Under ROC = {}".format(area))
