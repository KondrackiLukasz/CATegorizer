import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from model_evaluation import evaluate_model, plot_train_history
from kerastuner import RandomSearch


def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu',
                     input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(hp.Int('conv_1_units', min_value=32,
              max_value=256, step=32), (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(hp.Int('conv_2_units', min_value=32,
              max_value=256, step=32), (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_1_units', min_value=32,
              max_value=512, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


train_dir = 'data/train/'
val_dir = 'data/val/'
test_dir = 'data/test/'

# Normalize pixel values and apply data augmentation for training data
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

train_dataset = train_datagen.flow_from_directory(train_dir, target_size=(256, 256), class_mode='categorical',
                                                  batch_size=32, shuffle=True, seed=42)
val_dataset = train_datagen.flow_from_directory(val_dir, target_size=(256, 256), class_mode='categorical',
                                                batch_size=32, shuffle=True, seed=42)
test_dataset = train_datagen.flow_from_directory(test_dir, target_size=(256, 256), class_mode='categorical',
                                                 batch_size=32, shuffle=True, seed=42)

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # how many model configurations would you like to test?
    # how many trials per variation? (same model could perform differently)
    executions_per_trial=1,
    directory='model_dir',
    project_name='tune_model'
)

tuner.search(x=train_dataset,
             epochs=80,
             validation_data=val_dataset,
             callbacks=[tf.keras.callbacks.ModelCheckpoint(
                        'best_model_train2.h5', save_best_only=True, monitor='val_accuracy', mode='max')])

best_model = tuner.get_best_models(num_models=1)[0]


# Start of evaluation code

evaluate_model(best_model, train_datagen, train_dir, 'train')
evaluate_model(best_model, train_datagen, val_dir, 'val')
evaluate_model(best_model, train_datagen, test_dir, 'test')

# plot_train_history(history)
