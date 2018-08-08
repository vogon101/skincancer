from keras.callbacks import Callback
import matplotlib.pyplot as plt
from roc import classification_report, plot_roc

class GraphingCallback(Callback):

    def __init__(self, period, model, X_test, y_test, name):
        super(GraphingCallback, self).__init__()
        self.period = period
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.name = name
        self.accuracies = []
        self.areas = []

    def on_epoch_end(self, epoch, logs=None):

        if (epoch % self.period == 0):
            score = self.model.evaluate(self.X_test, self.y_test, verbose=2)
            self.accuracies.append(score[1])

            self.plot_all(epoch, True)

    def plot_all(self, epochs, append = True):
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5) * 1
        print(classification_report(self.y_test, y_pred))
        area = plot_roc(self.y_test, y_pred_proba,
                        title=self.name + ':' + str(epochs))
        print("Area Under ROC = {}".format(area))
        if append:
            self.areas.append(area)

        fig, ax1 = plt.subplots()

        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.plot(self.accuracies, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = "tab:blue"
        ax2.set_ylabel("AUC")
        ax2.plot(self.areas, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(self.name + ": Acc and AUC")
        plt.show()
