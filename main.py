from pathlib import Path

import keras
# import IPython
import numpy as np
import tensorflow as tf

from nc_eval import run_neural_compressor
from plots import show_distributions
from utils import LATENT_SIZE, run_model


# Load SavedModel into keras format, taken from:
# https://stackoverflow.com/questions/64945037/how-to-load-a-saved-tensorflow-model-and-evaluate-it
class LayerFromSavedModel(tf.keras.layers.Layer):
    def __init__(self, loaded):
        self.loaded = loaded
        super(LayerFromSavedModel, self).__init__()
        self.vars = loaded.variables

    def call(self, inputs):
        return self.loaded.signatures['serving_default'](inputs)


def main():
    model_path = Path("models/generators/new_generator")
    model = tf.keras.models.load_model(model_path)

    # Perform quantization on model
    run_neural_compressor(model)

    # Load model again in tensorflow format
    int8_model = tf.saved_model.load("int8_model.pb")

    # Convert model to keras format to be used at runtime.
    from keras.layers import Input
    input = keras.layers.Input(
        shape=(LATENT_SIZE,), dtype=np.float32)
    keras_i8_model = tf.keras.Model(input, LayerFromSavedModel(int8_model)(input))
    keras_i8_model.summary()

    # Run the models for performance measurements
    run_model(keras_i8_model, n_samples=100, n_batch_samples=256)
    run_model(model, n_samples=100, n_batch_samples=256)

    # Show distributions of both models overlayed to find out if there is an error after quantization.
    show_distributions(keras_i8_model, model)

    # profile_openvino(model, model_path)


if __name__ == "__main__":
    main()
