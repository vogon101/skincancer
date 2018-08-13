from keras.models import load_model
from keras import backend as K


class DLModel:

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
        print("[ WARN ] No architecture specified")

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
