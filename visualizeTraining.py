import matplotlib.pyplot as plt


def plot_history(history):
    # Retrieve training history
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()



val_loss = [1.7732, 1.7222, 1.8715, 2.3232, 3.9282, 5.0406, 5.0647, 7.3221, 7.3907, 7.2684, 8.2755, 9.0925, 10.8382,
            8.7446, 8.9583, 8.7228, 9.4760, 11.9575, 11.1167, 10.2612, 11.5609, 9.5599, 10.4437, 7.6110, 9.4570, 9.1552,
            9.4710, 11.5761, 13.5228, 10.9856, 12.7627, 13.1355, 11.9264, 11.8605, 11.0180, 13.3726, 13.7834, 12.5219,
            10.9917, 15.5810, 11.6936, 14.6993, 14.2401, 17.0867, 15.3068, 14.0868, 15.1783, 11.3070, 12.5206, 15.1891]
loss = [180.5356, 1.6936, 1.4380, 0.9952, 0.7056, 0.5710, 0.4193, 0.3162, 0.2855, 0.4204, 0.3414, 0.3394, 0.1647,
        0.2128, 0.1415, 0.1393, 0.1213, 0.1747, 0.1909, 0.1810, 0.1552, 0.2642, 0.3826, 0.2362, 0.1781, 0.2019, 0.0863,
        0.1624, 0.0888, 0.1153, 0.0608, 0.0557, 0.2531, 0.1078, 0.0423, 0.0366, 0.0468, 0.0258, 0.0554, 0.0586, 0.0945,
        0.0706, 0.0347, 0.0389, 0.0691, 0.0420, 0.1423, 0.0855, 0.0444, 0.0233]
accuracy = [0.1875, 0.2896, 0.4385, 0.6271, 0.7521, 0.8229, 0.8740, 0.8885, 0.9302, 0.9135, 0.9083, 0.9021, 0.9531,
            0.9448, 0.9583, 0.9594, 0.9740, 0.9688, 0.9719, 0.9615, 0.9708, 0.9615, 0.9646, 0.9448, 0.9604, 0.9688,
            0.9771, 0.9719, 0.9760, 0.9740, 0.9833, 0.9885, 0.9812, 0.9760, 0.9865, 0.9896, 0.9896, 0.9917, 0.9865,
            0.9896, 0.9885, 0.9854, 0.9906, 0.9927, 0.9885, 0.9906, 0.9812, 0.9823, 0.9875, 0.9948]
val_accuracy = [
    0.2250, 0.2917, 0.2833, 0.2750, 0.2750, 0.2333, 0.2250, 0.2333, 0.2167, 0.2417,
    0.3167, 0.3083, 0.2667, 0.2667, 0.2750, 0.2500, 0.2250, 0.2500, 0.2167, 0.1917,
    0.2833, 0.2500, 0.2167, 0.2333, 0.2500, 0.2500, 0.2000, 0.1833, 0.2083, 0.2583,
    0.2333, 0.2417, 0.2000, 0.2500, 0.2417, 0.2417, 0.2083, 0.2833, 0.2083, 0.2417,
    0.2417, 0.2583, 0.2750, 0.2583, 0.2667, 0.2500, 0.2500, 0.2833, 0.2417, 0.2500
]
history = {
    'loss': loss,
    'accuracy': accuracy,
    'val_loss': val_loss,
    'val_accuracy': val_accuracy
}

# Assuming 'history' variable contains the training history returned by model.fit()
plot_history(history)
