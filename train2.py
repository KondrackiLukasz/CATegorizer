from typing import Dict, Any
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import Precision, Recall
from model_evaluation import evaluate_model, plot_train_history

# Define directories
train_dir = 'data/train/'
val_dir = 'data/val/'
test_dir = 'data/test/'

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect'
)

# Load training and validation datasets
train_dataset = train_datagen.flow_from_directory(train_dir, target_size=(256, 256), class_mode='categorical',
                                                  batch_size=32, shuffle=True, seed=42)
val_dataset = train_datagen.flow_from_directory(val_dir, target_size=(256, 256), class_mode='categorical',
                                                batch_size=32, shuffle=True, seed=42)
test_dataset = train_datagen.flow_from_directory(test_dir, target_size=(256, 256), class_mode='categorical',
                                                 batch_size=32, shuffle=True, seed=42)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

# Define metrics
precision = Precision(name='precision')
recall = Recall(name='recall')

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', precision, recall])

# Prepare for training
num_epochs = 50
best_models = {'loss': (float('inf'), None), 'accuracy': (
    0, None), 'precision': (0, None), 'recall': (0, None), 'f1': (0, None)}
history_dict: Dict[str, Any] = {'loss': [], 'accuracy': [], 'val_loss': [
], 'val_accuracy': [], 'val_precision': [], 'val_recall': []}

for epoch in range(num_epochs):
    history = model.fit(train_dataset, epochs=1, validation_data=val_dataset)

    # Get metrics
    loss = history.history['val_loss'][0]
    accuracy = history.history['val_accuracy'][0]
    precision_value = history.history['val_precision'][0]
    recall_value = history.history['val_recall'][0]
    f1_value = 2 * ((precision_value * recall_value) /
                    (precision_value + recall_value + 1e-7))

    # Update history_dict
    history_dict['loss'].append(history.history['loss'][0])
    history_dict['accuracy'].append(history.history['accuracy'][0])
    history_dict['val_loss'].append(loss)
    history_dict['val_accuracy'].append(accuracy)
    history_dict['val_precision'].append(precision_value)
    history_dict['val_recall'].append(recall_value)

    # Update best models
    if loss < best_models['loss'][0]:
        best_models['loss'] = (loss, model.get_weights())
    if accuracy > best_models['accuracy'][0]:
        best_models['accuracy'] = (accuracy, model.get_weights())
    if precision_value > best_models['precision'][0]:
        best_models['precision'] = (precision_value, model.get_weights())
    if recall_value > best_models['recall'][0]:
        best_models['recall'] = (recall_value, model.get_weights())
    if f1_value > best_models['f1'][0]:
        best_models['f1'] = (f1_value, model.get_weights())

# Start of evaluation code
for metric, (value, weights) in best_models.items():
    model.set_weights(weights)
    model.save(f'best_model_{metric}.h5')
    evaluate_model(model, train_datagen, train_dir, f'train_{metric}')
    evaluate_model(model, train_datagen, val_dir, f'val_{metric}')
    evaluate_model(model, train_datagen, test_dir, f'test_{metric}')

# Plot training history
plot_train_history(history_dict)
