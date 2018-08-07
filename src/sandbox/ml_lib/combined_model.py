from keras.models import load_model
from keras import backend as K
from keras.applications import VGG16
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation, concatenate
from keras.models import  Model, Sequential
from keras import optimizers


class CombinedModel():
    def __init__(self, image_size=(128,128,3)):
        self.image_size = image_size
        self.img_rows = image_size[0]
        self.img_cols = image_size[1]
        self.img_channels = image_size[2]
        if K.image_dim_ordering() == 'th':
            self.input_shape = (self.img_channels, self.img_rows, self.img_cols)
        else:
            self.input_shape = (self.img_rows, self.img_cols, self.img_channels)
        print('Using Iput_shape = ', self.input_shape)
        self.architecture()

    def architecture(self):
        submodel_vgg = VGG16(
            include_top=False,
            input_shape=self.image_size,
            weights="imagenet"
        )

        for layer in submodel_vgg.layers:
            layer.trainable = False

        vgg_flatten = Flatten(input_shape=submodel_vgg.output_shape[1:])

        submodel_vgg_flat = Model(inputs=submodel_vgg.input, outputs=vgg_flatten(submodel_vgg.output))

        submodel_convolutional = Sequential()

        submodel_convolutional.add(Conv2D(32, (3,3), input_shape=(128,128,3)))
        submodel_convolutional.add(Activation("relu"))

        submodel_convolutional.add(MaxPooling2D((4,4)))

        submodel_convolutional.add(Conv2D(32, (3, 3)))
        submodel_convolutional.add(Activation("relu"))

        submodel_convolutional.add(MaxPooling2D())

        submodel_convolutional.add(Conv2D(64, (3, 3)))
        submodel_convolutional.add(Activation("relu"))

        submodel_convolutional.add(MaxPooling2D())

        submodel_convolutional.add(Flatten())

        print(submodel_convolutional.summary())

        submodel_classifier = Sequential()
        submodel_classifier.add(Flatten(input_shape=submodel_vgg.output_shape[1:]))
        submodel_classifier.add(Dense(128, activation="relu"))
        submodel_classifier.add(Dropout(0.5))
        submodel_classifier.add(Dense(1, activation="sigmoid"))

        combo_input = concatenate([submodel_convolutional.output, submodel_vgg_flat.output])
        print(combo_input.summary())

        self.model = Model(inputs=submodel_vgg.input, outputs=submodel_classifier(submodel_vgg.output))

        self.model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

    def fit(self, X_train, y_train, X_test, y_test, nb_epoch=10, batch_size=50):
        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], self.img_channels,
                                        self.img_rows, self.img_cols)
            X_test = X_test.reshape(X_test.shape[0], self.img_channels,
                                        self.img_rows, self.img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], self.img_rows,
                                        self.img_cols, self.img_channels)
            X_test = X_test.reshape(X_test.shape[0], self.img_rows,
                                        self.img_cols, self.img_channels)

        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=1, validation_data=(X_test, y_test))

        score = self.model.evaluate(X_test, y_test, verbose=2)
        print('Test score:', score[0])
        print('Test accuracy:', score[1]) # this is the one we care about
        return score

    def fit_generator(self, train_generator, validation_generator, nb_train, nb_validation, nb_epoch=10, batch_size=50, callbacks=None):
        self.model.fit_generator(
            train_generator,
            epochs=nb_epoch,
            steps_per_epoch= nb_train / batch_size,
            verbose=2,
            validation_data=validation_generator,
            validation_steps=nb_validation / batch_size,
            use_multiprocessing=False,
            callbacks = callbacks)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename='model_save.h5'):
        self.model.save(filename)
        print('Model saved to: ', filename)

    def load_model(self, filename):
        self.model = load_model(filename)
        print('Model Loaded: ', filename)

if __name__ == '__main__':
    pass
