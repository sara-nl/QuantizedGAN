from matplotlib import pyplot as plt


def show_distributions(i8model, source_model, batch_size=128, n_samples=100):
    values = []

    # Set plot size
    plt.rcParams["figure.figsize"] = (15, 12)

    fig, axs = plt.subplots(3)
    ax2s = []
    for ax in axs:
        ax2s.append(ax.twinx())

    handles = []
    for i, (model, color, f) in enumerate(
            [(source_model, "red", lambda x: x), (i8model, "blue", lambda x: x["sequential"])]):
        output = [f(model([EnergyGAN_input(batch_size)])) for i in range(n_samples)]
        print(type(model), type(output), len(output))
        output = np.array(output, dtype=np.float)
        output = output.reshape((-1, 51, 51, 25, 1))

        xsum = np.mean(output, axis=(0, 1, 2, 4))
        xdev = np.std(output, axis=(0, 1, 2, 4))
        ysum = np.mean(output, axis=(0, 2, 3, 4))
        ydev = np.std(output, axis=(0, 2, 3, 4))
        zsum = np.mean(output, axis=(0, 1, 3, 4))
        zdev = np.std(output, axis=(0, 1, 3, 4))

        values.append([xsum, ysum, zsum])

        for j, (s, d) in enumerate([(xsum, xdev), (ysum, ydev), (zsum, zdev)]):
            xs = np.arange(0, len(s))
            axs[j].set_ylabel("diff %")
            ax2s[j].set_ylabel("mean")
            axs[j].set_xlabel("10^(-10) m")
            handle = ax2s[j].bar(xs + 0.04 - (0.08 * i), s, yerr=d, ecolor="dark" + color, color=color)
        handles.append(handle)

    percentual_diff = lambda v1, v2: (v1 - v2) / v2 * 100
    axs[0].plot(percentual_diff(values[0][0], values[1][0]))
    axs[1].plot(percentual_diff(values[0][1], values[1][1]))
    handle, = axs[2].plot(percentual_diff(values[0][2], values[1][2]))
    handles.append(handle)

    fig.legend(handles, [f"INT8 Model", "Source Model", "Percentual difference"])
    fig.suptitle(f"Mean over axes (x, y, z), {batch_size * n_samples} samples")
    plt.savefig("distributions.svg")

def show_output(output):
    for i in range(len(output)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data = output[i, :, :, :, 0].reshape(25, 25, 25)
        z, x, y = data.nonzero()
        data = data[z, x, y]
        ax.scatter(x, y, z, c=data, alpha=1)
        plt.show()