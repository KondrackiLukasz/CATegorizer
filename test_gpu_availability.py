import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        print(f"Using GPU: {device}")
else:
    print("No GPU available. TensorFlow is running on CPU.")
