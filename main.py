import datetime
import math
import sys
from functools import partial
from pathlib import Path

import keras
from neural_compressor.experimental import common, Benchmark, Quantization
from scipy.stats import ks_2samp

# import IPython
import numpy as np
from matplotlib import pyplot as plt
from openvino.runtime import Core
import os
import tensorflow as tf
from tqdm import tqdm

from energyGAN.dataloader import get_data








def load_EnergyGAN(path):
    return tf.keras.models.load_model(path)


def EnergyGAN_input(nb_samples=10):
    noise = np.random.normal(0, 1, (nb_samples, 256))
    gen_aux = np.random.uniform(0.02, 5, (nb_samples, 1))  # E_p/100 vector
    generator_input = np.multiply(gen_aux, noise)
    return generator_input


def convert_openvino(model_path, precision="FP16", samples=10, force_overwrite=False, transform=False):
    gen_input = EnergyGAN_input(samples)

    print(f"Model path: {model_path}")

    # The paths of the source and converted models
    ir_path = Path("generators/saved_model").with_suffix(".xml")

    # Construct the command for Model Optimizer
    mo_command = f"""mo
                     --framework tf
                     --saved_model_dir "{model_path}"
                     --input_shape "[{str(gen_input.shape)[1:-1]}]"
                     --data_type {precision}
                     --output_dir "{model_path.parent}"
                     """
    if transform:
        mo_command += "--transform LowLatency2\n"

    mo_command = " ".join(mo_command.split())
    print("Model Optimizer command to convert TensorFlow to OpenVINO:")
    print(f"`{mo_command}`")

    # Run Model Optimizer if the IR model file does not exist
    if force_overwrite or not ir_path.exists():
        print("Exporting TensorFlow model to IR... This may take a few minutes.")
        a = os.system(mo_command)
        print(a)
    else:
        print(f"IR model {ir_path} already exists.")

    ie = Core()
    model = ie.read_model(model=ir_path, weights=ir_path.with_suffix(".bin"))
    ir_model = ie.compile_model(model=model, device_name="CPU")
    return ir_model


def run_model(model, n_samples=10, n_batch_samples=10):
    print(f"Processing {n_samples} samples on {type(model)}.")
    start_time = datetime.datetime.now()

    gen_input = EnergyGAN_input(nb_samples=n_batch_samples)
    for i in range(n_samples):
        model([gen_input])

    tot_time = datetime.datetime.now() - start_time
    print(
        f"{n_samples * n_batch_samples} samples processed in {tot_time} ({(n_samples * n_batch_samples) / tot_time.total_seconds()} samples/s)")
    return n_samples / tot_time.total_seconds()


# optional if Neural Compressor built-in metric could be used to do accuracy evaluation on model output in yaml
class CustomMetric(object):
    def __init__(self, exa, eya, eza):
        self.expected_xs = exa
        self.expected_ys = eya
        self.expected_zs = eza

        self.x_avgs = []
        self.y_avgs = []
        self.z_avgs = []

        self.x_vars = []
        self.y_vars = []
        self.z_vars = []
        self.n_updates = 0

    def update(self, predict, label):
        self.n_updates += 1
        self.x_avgs += [np.mean(predict, axis=(0, 1, 2, 4))]
        self.y_avgs += [np.mean(predict, axis=(0, 1, 3, 4))]
        self.z_avgs += [np.mean(predict, axis=(0, 2, 3, 4))]

        self.x_vars += [np.std(predict, axis=(0, 1, 2, 4))]
        self.y_vars += [np.std(predict, axis=(0, 1, 3, 4))]
        self.z_vars += [np.std(predict, axis=(0, 2, 3, 4))]

    def result(self):
        # Compute MSE distribution error
        self.x_avgs = np.array(self.x_avgs)
        self.y_avgs = np.array(self.y_avgs)
        self.z_avgs = np.array(self.z_avgs)

        x_avg = np.mean(self.x_avgs, axis=0)
        y_avg = np.mean(self.y_avgs, axis=0)
        z_avg = np.mean(self.z_avgs, axis=0)
        dist_error = (
                np.mean((x_avg - np.mean(self.expected_xs[0], axis=0)) ** 2) +
                np.mean((y_avg - np.mean(self.expected_ys[0], axis=0)) ** 2) +
                np.mean((z_avg - np.mean(self.expected_zs[0], axis=0)) ** 2)
        )

        # Compute MSE variance error
        self.x_vars = np.array(self.x_vars)
        self.y_vars = np.array(self.y_vars)
        self.z_vars = np.array(self.z_vars)

        x_var = np.mean(self.x_vars, axis=0)
        y_var = np.mean(self.y_vars, axis=0)
        z_var = np.mean(self.z_vars, axis=0)
        var_error = (
                np.mean((x_var - np.mean(self.expected_xs[1], axis=0)) ** 2) +
                np.mean((y_var - np.mean(self.expected_ys[1], axis=0)) ** 2) +
                np.mean((z_var - np.mean(self.expected_zs[1], axis=0)) ** 2)
        )

        print(dist_error, var_error)
        return dist_error + var_error

    def reset(self):
        self.x_avgs = []
        self.y_avgs = []
        self.z_avgs = []


def compute_distributions(model, train_dataset, save_distributions=True):
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()

    x_avgs = []
    y_avgs = []
    z_avgs = []
    x_vars = []
    y_vars = []
    z_vars = []
    print("Computing expected energy distributions")

    for batch, y in tqdm(train_dataset.batch(32)):
        predict = model(batch)

        x_avgs += [np.mean(predict, axis=(0, 1, 2, 4))]
        y_avgs += [np.mean(predict, axis=(0, 1, 3, 4))]
        z_avgs += [np.mean(predict, axis=(0, 2, 3, 4))]
        x_vars += [np.std(predict, axis=(0, 1, 2, 4))]
        y_vars += [np.std(predict, axis=(0, 1, 3, 4))]
        z_vars += [np.std(predict, axis=(0, 2, 3, 4))]

    x_avgs = np.array(x_avgs)
    y_avgs = np.array(y_avgs)
    z_avgs = np.array(z_avgs)
    x_vars = np.array(x_vars)
    y_vars = np.array(y_vars)
    z_vars = np.array(z_vars)

    if save_distributions:
        x_avg = np.mean(x_avgs, axis=0)
        y_avg = np.mean(y_avgs, axis=0)
        z_avg = np.mean(z_avgs, axis=0)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.errorbar(np.arange(len(x_avg)), x_avg, yerr=np.mean(x_vars, axis=0))
        plt.savefig("X_dist.png")

        plt.figure()
        plt.errorbar(np.arange(len(y_avg)), y_avg, yerr=np.mean(y_vars, axis=0))
        plt.savefig("Y_dist.png")

        plt.figure()
        plt.errorbar(np.arange(len(z_avg)), z_avg, yerr=np.mean(z_vars, axis=0))
        plt.savefig("Z_dist.png")

    return (x_avgs, x_vars), (y_avgs, y_vars), (z_avgs, z_vars)


def neural_compressor(model):
    #########################################
    # Load dataset into correct format
    #########################################
    dataset = []
    for i in range(1, 9):
        dataset_filename = f"dataset/EleEscan_RandomAngle_{i}_1.h5"
        data = get_data(dataset_filename)
        dataset.append(data)

    x, y, angle_train, ecal_train = zip(*dataset)
    x = np.concatenate(x)
    y = np.concatenate(y)
    angle_train = np.concatenate(angle_train)
    ecal_train = np.concatenate(ecal_train)

    latent_size = 256

    noise = np.random.normal(0, 1, (ecal_train.shape[0], latent_size - 2)).astype(np.float32)
    data = np.concatenate((ecal_train.reshape(-1, 1), angle_train.reshape(-1, 1), noise), axis=1)

    data_size = len(data)
    split = int(data_size * .7)
    train_dataset = tf.data.Dataset.from_tensor_slices((data[:split], y[:split]))
    test_dataset = tf.data.Dataset.from_tensor_slices((data[split:], y[split:]))

    #########################################
    # Compute (mean, var) distributions to use for accuracy
    #########################################
    xdist, ydist, zdist = compute_distributions(model, train_dataset)

    #########################################
    # Initialize model
    #########################################
    quantizer = Quantization(model)
    quantizer.model = model

    #########################################
    # Set dataloader in NC quantizer
    #########################################
    quantizer.calib_dataloader = common.DataLoader(train_dataset, batch_size=32)
    quantizer.eval_dataloader = common.DataLoader(test_dataset, batch_size=32)
    quantizer.metric = common.Metric(partial(CustomMetric, xdist, ydist, zdist))
    quantizer.cfg["tuning"]["accuracy_criterion"]["higher_is_better"] = False
    # quantizer.cfg["tuning"]["strategy"]["name"] = "exhaustive"
    # print("Config:\n", quantizer.cfg)


    quantized_graph: tf.Graph = quantizer.fit()

    quantized_graph.save("int8_model.pb")


def benchmark(model):
    train_dataset = None
    evaluator = Benchmark(model)

    # Specifically set some properties if you dont have a yaml
    evaluator.exp_benchmarker.framework = "tensorflow"
    evaluator.exp_benchmarker.b_dataloader = common.DataLoader(train_dataset)

    accs = []
    latencies = []
    for i in range(100):
        results = evaluator('./int8.pb')

        for mode, result in results.items():
            acc, batch_size, result_list = result
            latency = np.array(result_list).mean() / batch_size
            accs.append(acc)
            latencies.append(latency)

    avg_acc = sum(accs) / len(accs)
    avg_latency = sum(latencies) / len(latencies)
    avg_throughput = 1 / (sum(latencies) / len(latencies))
    print(f"Acc: {avg_acc}, Lat {avg_latency}, Throughput: {avg_throughput}")


def profile_openvino(model, model_path):
    perfs = []
    for precision in ["FP16", "FP32"]:
        print("Precision =", precision)
        ir_model = convert_openvino(
            model_path,
            precision,
            force_overwrite=True,
            transform=True
        )

        perf = run_model(model, n_samples=100)
        ov_perf = run_model(ir_model, n_samples=100)
        perfs.append(" | ".join(str(a) for a in [perf, ov_perf]))
    print("Native | OpenVINO")
    for perf in perfs:
        print(perf)


class LayerFromSavedModel(tf.keras.layers.Layer):
    def __init__(self, loaded):
        self.loaded = loaded
        super(LayerFromSavedModel, self).__init__()
        self.vars = loaded.variables

    def call(self, inputs):
        return self.loaded.signatures['serving_default'](inputs)


from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, Dropout, BatchNormalization, Activation)


def main():
    model_path = Path("generators/new_generator")
    model = load_EnergyGAN(model_path)

    # Perform quantization on model
    neural_compressor(model)

    # Load model again in tensorflow format
    int8_model = tf.saved_model.load("int8_model.pb")

    # Convert model to keras format to be used at runtime.
    input = keras.layers.Input(
        shape=(256,), dtype=np.float32)
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
