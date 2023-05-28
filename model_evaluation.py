import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

figure_counter = 1  # Counter for the figures


def plot_classification_report(classificationReport, title='Classification report', cmap='RdYlGn'):
    df = pd.DataFrame(classificationReport).transpose()

    # If 'support' column exists, drop it
    if 'support' in df.columns:
        df = df.drop('support', axis=1)

    df = df.round(2)  # round to two decimal points

    # Separate heatmap of classes from accuracy, macro avg, and weighted avg
    classes_heatmap = df.iloc[:-3, :]
    metrics_heatmap = df.iloc[-3:, :]

    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

    sns.heatmap(classes_heatmap, annot=True, cmap=cmap, ax=axes[0])
    axes[0].set_title(title)

    sns.heatmap(metrics_heatmap, annot=True, cmap=cmap, ax=axes[1])
    axes[1].set_title("Metrics: Accuracy, Macro Avg, Weighted Avg")

    # Relocate labels to the top of the chart
    axes[0].xaxis.tick_top()
    axes[0].xaxis.set_label_position('top')

    plt.tight_layout()
    # plt.show()

    global figure_counter  # Access the global counter
    plt.savefig(f'Figure_{figure_counter}.png')  # Save the figure
    plt.close()  # Close the current figure
    figure_counter += 1  # Increment the figure counter


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    global figure_counter  # Access the global counter
    plt.savefig(f'Figure_{figure_counter}.png')  # Save the figure
    plt.close()  # Close the current figure
    figure_counter += 1  # Increment the figure counter


def plot_train_history(history):
    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    global figure_counter  # Access the global counter
    plt.savefig(f'Figure_{figure_counter}.png')  # Save the figure
    plt.close()  # Close the current figure
    figure_counter += 1  # Increment the figure counter


def evaluate_model(model, datagen, dataset_dir, dataset_name):
    dataset = datagen.flow_from_directory(dataset_dir, target_size=(
        256, 256), class_mode='categorical', batch_size=32, shuffle=False, seed=42)

    y_actual_list = []
    y_test_list = []

    for i in range(len(dataset)):
        y_actual_list.extend(np.argmax(dataset[i][1], axis=1))
        y_test_list.extend(np.argmax(model.predict(dataset[i][0]), axis=1))

    y_actual = np.array(y_actual_list)
    y_test = np.array(y_test_list)

    target_names = list(dataset.class_indices.keys())
    print(f"Evaluating on {dataset_name} set")
    report = classification_report(
        y_actual, y_test, target_names=target_names, output_dict=True)
    plot_classification_report(
        report, title=f'Classification Report for {dataset_name}')

    cm = confusion_matrix(y_actual, y_test)
    plot_confusion_matrix(cm, classes=target_names,
                          title=f'Confusion Matrix for {dataset_name}')

    print('Accuracy: ', accuracy_score(y_actual, y_test))
