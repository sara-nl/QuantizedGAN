import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from utils import LATENT_SIZE, run_model, run_model_openvino


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
    parser.add_argument("--nc_model", metavar="o", type=str,
                        help="Output dir for neural compressor, only used when --nc=true",
                        default="models/int8_model")
    parser.add_argument("--profile", metavar="p", type=str, help="Turn on profiling for onnx/default/nc.", default="")
    parser.add_argument("--model", metavar="m", type=str, help="Input model to be used.",
                        default="models/generators/new_generator")
    parser.add_argument("--batch_size", type=int, help="Batch size.",
                        default=256)
    parser.add_argument("--dtype", type=str, help="Neural compressor dtype",
                        default="int8")
    parser.add_argument("--onnx_model", type=str, help="ONNX Model",
                        default="models/model.onnx")
    parser.add_argument("--n_samples", type=int, help="N Samples to run for profiling",
                        default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    default_model = tf.keras.models.load_model(model_path)
    default_model.summary()
    default_model.get_layer(index=1).summary()
    # Perform quantization on model
    if args.nc:
        from nc_eval import run_neural_compressor
        run_neural_compressor(default_model, output=args.nc_model, batch_size=args.batch_size, dtype=args.dtype)

    if args.profile != "":
        models = args.profile.split(",")
        print(f"Running on tensorflow {tf.__version__}")
        tf.profiler.experimental.start("logdir")
        for model in models:
            if model == "nc":
                # Load model again in tensorflow format
                int8_model = tf.saved_model.load(args.nc_model)
                # Convert model to keras format to be used at runtime.
                # https://github.com/tensorflow/tensorflow/issues/42425
                from keras.layers import Input
                input_shape = keras.layers.Input(
                    shape=(LATENT_SIZE,), dtype=np.float32)
                keras_i8_model = tf.keras.Model(input_shape, LayerFromSavedModel(int8_model)(input_shape))

                # Run the models for performance measurements
                run_model(keras_i8_model, n_samples=args.n_samples, n_batch_samples=args.batch_size)
            elif model == "default":
                run_model(default_model, n_samples=args.n_samples, n_batch_samples=args.batch_size)
            elif model == "onnx":
                import onnxruntime
                so = onnxruntime.SessionOptions()
                so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = onnxruntime.InferenceSession(args.onnx_model, so, providers=[
                    'OpenVINOExecutionProvider'], provider_options=[{"device_type": "CPU_FP32"}])
                run_model_openvino(session, n_samples=args.n_samples, n_batch_samples=args.batch_size)
        tf.profiler.experimental.stop()
    # Show distributions of both models overlayed to find out if there is an error after quantization.
    # show_distributions(keras_i8_model, model)


if __name__ == "__main__":
    main()
