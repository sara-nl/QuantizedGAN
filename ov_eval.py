import os
from pathlib import Path

from openvino.runtime import Core

from utils import generate_random_input, run_model


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


def convert_openvino(model_path, precision="FP16", samples=10, force_overwrite=False, transform=False):
    gen_input = generate_random_input(samples)

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
