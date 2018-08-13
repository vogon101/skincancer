from ml_lib.dlmodel import DLModel
from keras.applications import VGG16
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation, concatenate, Input
from keras.models import  Model, Sequential
from keras import optimizers


class CombinedModelV2(DLModel):

    def architecture(self):

        submodel_input = Input((128,128,3), name="image_in")

        submodel_vgg = VGG16(
            include_top=False,
            input_shape=self.image_size,
            weights="imagenet"
        )

        for layer in submodel_vgg.layers:
            layer.trainable = False

        submodel_vgg = submodel_vgg(submodel_input)
        submodel_vgg = Conv2D(128, (2,2), name="submodel_vgg_end_conv")(submodel_vgg)
        submodel_vgg = Activation("relu")(submodel_vgg)
        submodel_vgg = Flatten()(submodel_vgg)
        submodel_vgg = Dense(64, activation="relu", name="submodel_vgg_out")(submodel_vgg)
        submodel_vgg = Dropout(0.5)(submodel_vgg)

        submodel_convolutional = Sequential()(submodel_input)

        submodel_convolutional = Conv2D(32, (3, 3))(submodel_convolutional)
        submodel_convolutional = Activation("relu")(submodel_convolutional)

        submodel_convolutional = MaxPooling2D((2, 2))(submodel_convolutional)

        submodel_convolutional = Conv2D(64, (5, 5))(submodel_convolutional)
        submodel_convolutional = Activation("relu")(submodel_convolutional)

        submodel_convolutional = MaxPooling2D((3, 3))(submodel_convolutional)

        submodel_convolutional = Conv2D(64, (5, 5))(submodel_convolutional)
        submodel_convolutional = Activation("relu")(submodel_convolutional)

        submodel_convolutional = MaxPooling2D((3, 3))(submodel_convolutional)

        submodel_convolutional = Flatten()(submodel_convolutional)
        submodel_convolutional = Dense(64, activation="relu", name="submodel_convo_out")(submodel_convolutional)
        submodel_convolutional = Dropout(0.5)(submodel_convolutional)

        submodel_ouput = concatenate([submodel_vgg, submodel_convolutional])

        submodel_classifier = Dense(128, activation="relu")(submodel_ouput)
        submodel_classifier = Dropout(0.5)(submodel_classifier)
        submodel_classifier = Dense(1, activation="sigmoid")(submodel_classifier)

        self.model = Model(inputs = submodel_input, outputs = submodel_classifier)

        print(self.model.summary())

        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                           metrics=['accuracy'])



if __name__ == '__main__':
    model = CombinedModelV2()
