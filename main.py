import argparse
from pathlib import Path

import keras
# import IPython
import numpy as np
import tensorflow as tf

from nc_eval import run_neural_compressor
from ov_eval import profile_openvino
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


def parse_args():
    parser = argparse.ArgumentParser(description="Optimizations for CERN's AngleGAN")
    parser.add_argument("--nc", metavar="n", type=bool, help="Perform quantization with Intel Neural Compressor",
                        default=False)
    parser.add_argument("--openvino", metavar="v", type=bool, help="Perform model optimization with OpenVINO",
                        default=False)
    parser.add_argument("--nc_output", metavar="o", type=str,
                        help="Output dir for neural compressor, only used when --nc=true",
                        default="models/int8_model")
    parser.add_argument("--profile", metavar="p", type=bool, help="Turn on profiling.", default=False)
    parser.add_argument("--model", metavar="m", type=str, help="Input model to be used.",
                        default="models/generators/new_generator")

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    model = tf.keras.models.load_model(model_path)

    # Perform quantization on model
    if args.nc:
        run_neural_compressor(model, output=args.nc_output)

    if args.profile:
        # Load model again in tensorflow format
        int8_model = tf.saved_model.load(args.nc_output)

        # Convert model to keras format to be used at runtime.
        from keras.layers import Input
        input_shape = keras.layers.Input(
            shape=(LATENT_SIZE,), dtype=np.float32)
        keras_i8_model = tf.keras.Model(input_shape, LayerFromSavedModel(int8_model)(input_shape))
        keras_i8_model.summary()

        # Run the models for performance measurements
        run_model(keras_i8_model, n_samples=100, n_batch_samples=256)
        run_model(model, n_samples=100, n_batch_samples=256)

    # Show distributions of both models overlayed to find out if there is an error after quantization.
    # show_distributions(keras_i8_model, model)

    if args.openvino:
        profile_openvino(model, model_path)


if __name__ == "__main__":
    main()
