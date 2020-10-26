from matplotlib import pyplot as plt
import pandas as pd


def _plot_accuracy(data):
    acc = data['accuracy']
    val_acc = data['val_accuracy']
    loss = data['loss']
    val_loss = data['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def plot_from_history(history):
    _plot_accuracy(history.history)


def plot_from_log(log):
    _plot_accuracy(pd.read_csv(log))
