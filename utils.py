import h5py
import GANutils as gan

import datetime

import numpy as np
from tqdm import tqdm

LATENT_SIZE = 256


def generate_random_input(nb_samples=10, latent_size=LATENT_SIZE):
    noise = np.random.normal(0.1, 1, (nb_samples, latent_size))
    sampled_energies = np.random.uniform(0.1, 5, (nb_samples, 1))
    generator_ip = np.multiply(sampled_energies, noise)
    return generator_ip


def run_model(model, n_samples=10, n_batch_samples=10):
    print(f"Processing {n_samples} samples on {type(model)}.")
    model([generate_random_input(nb_samples=n_batch_samples)])
    gen_input = generate_random_input(nb_samples=n_batch_samples)

    start_time = datetime.datetime.now()
    for i in range(n_samples):
        model([gen_input])

    tot_time = datetime.datetime.now() - start_time
    print(
        f"{n_samples * n_batch_samples} samples processed in {tot_time} ({(n_samples * n_batch_samples) / tot_time.total_seconds()} samples/s)")
    return n_samples / tot_time.total_seconds()


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


# Dataloader taken from;
# https://github.com/svalleco/3Dgan/blob/Anglegan/keras/AngleTrain3dGAN.py
# get data for training
def GetDataAngle(datafile, xscale=1, xpower=1, yscale=100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1):
    print('Loading Data from .....', datafile)
    f = h5py.File(datafile, 'r')
    X = np.array(f.get('ECAL')) * xscale
    Y = np.array(f.get('energy')) / yscale
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))
    indexes = np.where(ecal > 10.0)
    X = X[indexes]
    Y = Y[indexes]
    if angtype in f:
        ang = np.array(f.get(angtype))[indexes]
    else:
        ang = gan.measPython(X)
    X = np.expand_dims(X, axis=daxis)
    ecal = ecal[indexes]
    ecal = np.expand_dims(ecal, axis=daxis)
    if xpower != 1.:
        X = np.power(X, xpower)
    return X, Y, ang, ecal
