from keras.preprocessing.image import ImageDataGenerator
from ml_lib.cnn_model import CNN
from ml_lib.moleimages import MoleImages

train_data_dir = 'data_scaled/'
validation_data_dir = 'data_scaled_validation/'
nb_train_samples = 873
nb_validation_samples = 96
epochs = 20
batch_size = 50


train_datagen = ImageDataGenerator(
    vertical_flip=True,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='binary'
)


#model = mycnn.fit_generator(train_generator,validation_generator,
#    nb_train_samples, nb_validation_samples, epochs, batch_size)

mimg = MoleImages()

X_test, y_test = mimg.load_test_images('data_scaled_validation/benign', 'data_scaled_validation/malign')

X_train, y_train = mimg.load_test_images('data_scaled/benign', 'data_scaled/malign')

mycnn = CNN()

mycnn.fit(
    X_train, y_train,
    X_test, y_test,
    epochs, batch_size
)

mycnn.save_model("models/model_1_test.h5")