import matplotlib.pyplot as plt
import numpy as np
import keras


class DataVisualized(keras.callbacks.Callback):
    def __init__(self, encoder, x_data, y_data, value_dropout_count=0):
        super()
        fig, ax = plt.subplots()
        count = len(y_data)
        self.ax = ax
        self.x_data = x_data
        self.y_data = y_data
        self.value_dropout_count = value_dropout_count
        self.encoder = encoder
        self.scatter = ax.scatter(x=np.zeros((count, 1)), y=np.zeros((count, 1)), c=y_data, cmap=plt.cm.jet)
        plt.colorbar(self.scatter)
        self.epoch_c = 0
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_c += 1
        # print("\n\nepoch: ", self.epoch_c)
        data = self.encoder.predict(self.x_data)
        self.ax.set_xlim(np.sort(data[:, 0])[(self.value_dropout_count - 1)],
                    np.sort(data[:, 0])[(-self.value_dropout_count)])
        self.ax.set_ylim(np.sort(data[:, 1])[(self.value_dropout_count - 1)],
                    np.sort(data[:, 1])[(-self.value_dropout_count)])
        self.scatter.set_offsets(data)

        if self.epoch_c % 5:
            pass
        plt.pause(0.5)
        pass

class DataTester(keras.callbacks.Callback):
    def __init__(self, x_data, period=1):
        super()
        self.x_data = x_data
        self.period = period
        self.epoch_c = 0
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_c += 1
        if self.epoch_c % self.period:
            loss = self.model.evaluate(self.x_data, self.x_data, batch_size=32, verbose=1)
            print('\n\nevaluation result: ', loss)

        pass