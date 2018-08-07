from ml_lib.vgg_model import VGGModel
from ml_lib.moleimages import MoleImages
from ml_lib.roc import plot_roc
import tensorflow as tf
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator

config = tf.ConfigProto(
    #device_count = {"GPU": 0}
)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

train_data_dir = 'data_scaled/'
validation_data_dir = 'data_scaled_validation/'
model_save_path = "models/model_3_vgg_DA.h5"
nb_train_samples = 1853
nb_validation_samples = 204
batch_size = 32
epochs_per_test = 3
nb_total_tests = 30

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


my_model = VGGModel()

my_model.load_model()

print(my_model.model.summary())

print("Starting training for {} epochs".format(nb_total_tests * epochs_per_test))

accuracies = []
areas = []
for i in range(nb_total_tests):

    history = my_model.fit_generator(
        train_generator,
        validation_generator,
        nb_train_samples,
        nb_validation_samples,
        epochs_per_test,
        batch_size
    )

    score = my_model.model.evaluate(X_test, y_test, verbose=2)
    accuracies.append(score[1])

    print(score)

    print ("Model achieved accuracy of {}".format(score[1]))

    my_model.save_model(model_save_path)

    y_pred_proba = my_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5) * 1
    print(classification_report(y_test, y_pred))
    area = plot_roc(y_test, y_pred_proba, title='ROC Curve VGG transfer model:' + str((i + 1) * epochs_per_test))
    print("Area_ROC = {}".format(area))
    areas.append(area)

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(accuracies, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("AUC")
    ax2.plot(areas, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Accuracy and AUC over time")
    plt.show()

