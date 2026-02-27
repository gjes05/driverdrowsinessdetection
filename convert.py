import tensorflow as tf

model = tf.keras.models.load_model('my_model.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Do NOT use DEFAULT optimization — it can upgrade op versions
# Convert with no quantization to preserve op compatibility
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with open('my_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Done. Size:", len(tflite_model) / 1024 / 1024, "MB")