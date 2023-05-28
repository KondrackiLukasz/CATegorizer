import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from model_evaluation import evaluate_model, plot_train_history

train_dir = 'data/train/'
val_dir = 'data/val/'
test_dir = 'data/test/'

train_datagen = ImageDataGenerator()

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

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(
                        'best_model_train1.h5', save_best_only=True, monitor='val_accuracy', mode='max')])

# Start of evaluation code
model = tf.keras.models.load_model('best_model_train1.h5')

evaluate_model(model, train_datagen, train_dir, 'train')
evaluate_model(model, train_datagen, val_dir, 'val')
evaluate_model(model, train_datagen, test_dir, 'test')

plot_train_history(history)
