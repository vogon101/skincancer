from keras.applications import VGG16
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation, concatenate, Input
from keras.models import Sequential, Model

from ml_lib.combined_model_v2 import CombinedModelV2

submodel_input = Input((128,128,3), name="image_in")

submodel_convolutional = Sequential()(submodel_input)

submodel_convolutional = Conv2D(32, (3,3))(submodel_convolutional)
submodel_convolutional = Activation("relu")(submodel_convolutional)

submodel_convolutional = MaxPooling2D((2,2))(submodel_convolutional)

submodel_convolutional = Conv2D(64, (5,5))(submodel_convolutional)
submodel_convolutional = Activation("relu")(submodel_convolutional)

submodel_convolutional = MaxPooling2D((3,3))(submodel_convolutional)

submodel_convolutional = Conv2D(64, (5,5))(submodel_convolutional)
submodel_convolutional = Activation("relu")(submodel_convolutional)

submodel_convolutional = MaxPooling2D((3,3))(submodel_convolutional)

submodel_convolutional = Flatten()(submodel_convolutional)
submodel_convolutional = Dense(64, activation="relu", name="submodel_convo_out")(submodel_convolutional)
submodel_convolutional = Dropout(0.5)(submodel_convolutional)

model = Model(inputs=submodel_input, outputs=submodel_convolutional)

print(model.summary())
