from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from ml_lib.dlmodel import DLModel


class CNN(DLModel):

    def architecture(self):
        self.nb_filters = 32
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.model = Sequential()

        # Input Shape (128,128,3)
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(3, 3), input_shape=self.input_shape))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # self.model.add(Dropout(0.5))

        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(3, 3)))
        self.model.add(Activation('relu'))
        # Shape: 61x61x32

        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        # Shape: 30x30x32

        #self.model.add(Conv2D(self.nb_filters * 2, self.kernel_size))
        #self.model.add(Activation('relu'))
        # Shape: 28x28x64

        #self.model.add(MaxPooling2D(pool_size=self.pool_size))
        # Shape: 14x14x64

        self.model.add(Flatten())
        print('Model flattened out to ', self.model.output_shape)
        # Shape: 12544

        # now start a typical neural network
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        # Shape: 64

        self.model.add(Dropout(0.2))

        print('Using 2 Classes')
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        loss = 'binary_crossentropy'
        self.model.compile(loss=loss,
                           optimizer='adam',
                           metrics=['accuracy'])


if __name__ == '__main__':
    pass
