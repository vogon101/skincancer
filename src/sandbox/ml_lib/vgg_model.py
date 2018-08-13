from ml_lib.dlmodel import DLModel
from keras.applications import VGG16
from keras.layers import Dropout, Flatten, Dense
from keras.models import  Model, Sequential
from keras import optimizers


class VGGModel(DLModel):

    def architecture(self):
        submodel_vgg = VGG16(
            include_top=False,
            input_shape=self.image_size,
            weights="imagenet"
        )

        for layer in submodel_vgg.layers:
            layer.trainable = False

        submodel_classifier = Sequential()
        submodel_classifier.add(Flatten(input_shape=submodel_vgg.output_shape[1:]))
        submodel_classifier.add(Dense(128, activation="relu"))
        submodel_classifier.add(Dropout(0.5))
        submodel_classifier.add(Dense(1, activation="sigmoid"))

        self.model = Model(inputs=submodel_vgg.input, outputs=submodel_classifier(submodel_vgg.output))

        self.model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])


if __name__ == '__main__':
    pass
