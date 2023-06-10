import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

data_path = 'data/raw/'
train_dir = 'data/train/'
val_dir = 'data/val/'
test_dir = 'data/test/'

datagen = ImageDataGenerator()

train_dataset = datagen.flow_from_directory(train_dir, target_size=(256, 256), class_mode='categorical',
                                            batch_size=32)
val_dataset = datagen.flow_from_directory(val_dir, target_size=(256, 256), class_mode='categorical',
                                          batch_size=32)
test_dataset = datagen.flow_from_directory(test_dir, target_size=(256, 256), class_mode='categorical',
                                           batch_size=32)

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
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(
                        'best_model2.h5', save_best_only=True, monitor='val_accuracy')])

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)

# Save the best model
model.save('best_model2.h5')
