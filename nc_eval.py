# optional if Neural Compressor built-in metric could be used to do accuracy evaluation on model output in yaml
import pprint
from functools import partial

import numpy as np
from neural_compressor import Benchmark
from neural_compressor.experimental import Quantization, common
from neural_compressor.model.model import tf

from utils import compute_distributions, LATENT_SIZE, GetDataAngle


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
        process = lambda x, y, z: (
            np.mean(np.array(x), axis=0),
            np.mean(np.array(y), axis=0),
            np.mean(np.array(z), axis=0)
        )

        # Compute MSE distribution error
        x_avg, y_avg, z_avg = process(self.x_avgs, self.y_avgs, self.z_avgs)
        dist_error = (
                np.mean((x_avg - np.mean(self.expected_xs[0], axis=0)) ** 2) +
                np.mean((y_avg - np.mean(self.expected_ys[0], axis=0)) ** 2) +
                np.mean((z_avg - np.mean(self.expected_zs[0], axis=0)) ** 2)
        )

        # Compute MSE variance error
        x_var, y_var, z_var = process(self.x_vars, self.y_vars, self.z_vars)
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


def benchmark(model):
    train_dataset = None
    evaluator = Benchmark(model)

    # Specifically set some properties if you don't have a yaml
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


def run_neural_compressor(model, output="models/int8"):
    #########################################
    # Load dataset into correct format
    #########################################
    dataset = []
    for i in range(1, 9):
        dataset_filename = f"dataset/EleEscan_RandomAngle_{i}_1.h5"
        data = GetDataAngle(dataset_filename)
        dataset.append(data)

    x, y, angle_train, ecal_train = zip(*dataset)
    x = np.concatenate(x)
    y = np.concatenate(y)
    angle_train = np.concatenate(angle_train)
    ecal_train = np.concatenate(ecal_train)

    noise = np.random.normal(0, 1, (ecal_train.shape[0], LATENT_SIZE - 2)).astype(np.float32)
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

    #########################################
    # Set some specific tuning parameters
    #########################################
    quantizer.cfg["tuning"]["accuracy_criterion"]["higher_is_better"] = False

    print("Using this quantization configuration:")
    pprint.pprint(quantizer.cfg)

    quantized_graph: tf.Graph = quantizer.fit()

    quantized_graph.save(output)
