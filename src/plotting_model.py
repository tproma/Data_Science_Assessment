import matplotlib.pyplot as plt

def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label = 'Training accuracy')
    plt.plot(epochs, val_acc, 'b', label = 'Test accuracy')
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Test loss')
    plt.title("Training and Test loss")
    plt.legend()

    plt.show()

