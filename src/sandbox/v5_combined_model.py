from ml_lib.combined_model import CombinedModel
from ml_lib.moleimages import MoleImages
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from ml_lib.callbacks import GraphingCallback



config = tf.ConfigProto(
    #device_count = {"GPU": 0}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

train_data_dir = 'data_scaled/'
validation_data_dir = 'data_scaled_validation/'
model_save_path = "models/model_5_combined_DA.h5"
nb_train_samples = 1853
nb_validation_samples = 204
batch_size = 32
nb_epochs = 150

mimg = MoleImages()
X_test, y_test = mimg.load_test_images('data_scaled_validation/benign', 'data_scaled_validation/malign')

train_datagen = ImageDataGenerator(
    rotation_range=180,
    vertical_flip=True,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(128,128),
    batch_size=batch_size,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(128,128),
    batch_size=batch_size,
    class_mode="binary"
)

#my_model.load_model()


my_model = CombinedModel()

my_model.load_model(model_save_path)

print(my_model.model.summary())

#plot_model(my_model.model, to_file='model.png')

print("Starting training for {} epochs".format(nb_epochs))

grapher = GraphingCallback(5, my_model, X_test, y_test, "Combined Model")

best_model_VA = ModelCheckpoint('models/auto/v5_bm_acc_{epoch:02d}_{val_acc:.2f}.h5',monitor='val_acc',
                                mode = 'max', verbose=1, save_best_only=True)

callbacks = [
    grapher,
    best_model_VA
]

history = my_model.fit_generator(
        train_generator,
        validation_generator,
        nb_train_samples,
        nb_validation_samples,
        nb_epochs,
        batch_size,
        callbacks
    )

grapher.plot_all(nb_epochs, False)

my_model.save_model(model_save_path)
