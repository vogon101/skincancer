from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

class CNN():
    def __init__(self, image_size=(128,128,3), nb_classes=2,
                nb_filters=32, kernel_size=(3,3), pool_size=(2,2)):
        self.image_size = image_size
        self.img_rows = image_size[0]
        self.img_cols = image_size[1]
        self.img_channels = image_size[2]
        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        if K.image_dim_ordering() == 'th':
            self.input_shape = (self.img_channels, self.img_rows, self.img_cols)
        else:
            self.input_shape = (self.img_rows, self.img_cols, self.img_channels)
        print('Using Iput_shape = ', self.input_shape)
        self.architecture()

    def architecture(self):
        self.model = Sequential()

        #Input Shape (128,128,3)
        self.model.add(Conv2D(self.nb_filters,
                            (self.kernel_size[0], self.kernel_size[1]),
                            padding='valid',
                            input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        #Shape: 128x128x3

        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        #Shape: 63x63x32

        #self.model.add(Dropout(0.5))

        self.model.add(Conv2D(self.nb_filters,
                            (self.kernel_size[0], self.kernel_size[1])))
        self.model.add(Activation('relu'))
        #Shape: 61x61x32

        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        #Shape: 30x30x32

        self.model.add(Conv2D(self.nb_filters*2,
                            (self.kernel_size[0], self.kernel_size[1])))
        self.model.add(Activation('relu'))
        #Shape: 28x28x64

        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        #Shape: 14x14x64

        self.model.add(Flatten())
        print('Model flattened out to ', self.model.output_shape)
        #Shape: 12544

        # now start a typical neural network
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        #Shape: 64

        self.model.add(Dropout(0.5))

        if self.nb_classes>2:
            self.model.add(Dense(self.nb_classes))
            self.model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-9
            loss = 'categorical_crossentropy'
        else:
            print('Using 2 Classes')
            self.model.add(Dense(1))
            self.model.add(Activation('sigmoid'))
            loss = 'binary_crossentropy'
        self.model.compile(loss=loss,
                      optimizer='adadelta',
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

    def fit_generator(self, train_generator, validation_generator, n_samples,
                        n_validation, nb_epoch=10, batch_size=50, callbacks=None):
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=n_samples//batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=n_validation//batch_size,
            use_multiprocessing=True,
            workers=1,
            callbacks = callbacks)

        #score = self.model.evaluate(X_test, Y_test, verbose=0)
        #print('Test score:', score[0])
        #print('Test accuracy:', score[1]) # this is the one we care about
        return self.model

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
