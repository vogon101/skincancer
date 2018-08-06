from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

nb_filters = 32
kernel_size = (3,3)
input_shape = (128, 128, 3)
pool_size = (2,2)
nb_classes = 2


model = Sequential()
model.add(Conv2D(nb_filters,
                    (kernel_size[0], kernel_size[1]),
                    padding='valid',
                    input_shape=input_shape))

print(model.output_shape)

#first conv. layer (keep layer)
model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers
model.add(MaxPooling2D(pool_size=pool_size))

print(model.output_shape)
#model.add(Dropout(0.5))

model.add(Conv2D(nb_filters,
                    (kernel_size[0], kernel_size[1]))) #2nd conv. layer (keep layer)
model.add(Activation('relu'))

print(model.output_shape)

model.add(MaxPooling2D(pool_size=pool_size))

print(model.output_shape)

model.add(Conv2D(nb_filters*2,
                    (kernel_size[0], kernel_size[1]))) #2nd conv. layer (keep layer)
model.add(Activation('relu'))

print(model.output_shape)

model.add(MaxPooling2D(pool_size=pool_size))

print(model.output_shape)

model.add(Flatten()) # necessary to flatten before going into conventional dense layer (keep layer)
print('Model flattened out to ', model.output_shape)

# now start a typical neural network
model.add(Dense(64)) # (only) 32 neurons in this layer, really?  (keep layer)
model.add(Activation('relu'))

print(model.output_shape)

model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

print(model.output_shape)

if nb_classes>2:
    model.add(Dense(nb_classes)) # 10 final nodes (one for each class) (keep layer)
    model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-9
    loss = 'categorical_crossentropy'
else:
    print('Using 2 Classes')
    model.add(Dense(1)) # 10 final nodes (one for each class) (keep layer)
    model.add(Activation('sigmoid')) # keep softmax at end to pick between classes 0-9

    print(model.output_shape)

    loss = 'binary_crossentropy'
model.compile(loss=loss,
              optimizer='adadelta',  #adadelta
              metrics=['accuracy'])