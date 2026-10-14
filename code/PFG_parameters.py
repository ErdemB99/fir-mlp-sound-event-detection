
from pathlib import Path
import random

import numpy as np
import torch


def readParameters(filename):
    """
    Read network parameters from inputParameters.txt.

    Expected format:
        3       Number of layers
        128     Input neurons
        64      Hidden neurons
        11      Output neurons
        5       First FIR filter length
        3       Second FIR filter length
        0.01    Learning rate
        0.9     Momentum
    """

    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(
            f"Parameter file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        parameters = [
            line.strip()
            for line in f
            if line.strip()
            and not line.lstrip().startswith("#")
        ]

    return parameters


def load_config(filename):
    """
    Load and validate the FIR-MLP configuration.

    Returns:
        Dictionary containing architecture
        and training hyperparameters.
    """

    params = readParameters(filename)

    if len(params) != 8:
        raise ValueError(
            "Expected exactly 8 parameters in "
            "inputParameters.txt, "
            f"but received {len(params)}."
        )

    # ---------------------------------------
    # Network architecture
    # ---------------------------------------

    num_layers = int(params[0])

    num_neurons = [
        int(params[1]),
        int(params[2]),
        int(params[3])
    ]

    # ---------------------------------------
    # Temporal FIR filter lengths
    # ---------------------------------------

    fir_lengths = [
        int(params[4]),
        int(params[5])
    ]

    # ---------------------------------------
    # Training hyperparameters
    # ---------------------------------------

    learning_rate = float(params[6])
    momentum = float(params[7])

    # ---------------------------------------
    # Validate architecture
    # ---------------------------------------

    if num_layers != 3:
        raise ValueError(
            "The current FIR-MLP implementation "
            "requires exactly 3 layers."
        )

    if any(n <= 0 for n in num_neurons):
        raise ValueError(
            "All neuron counts must be positive."
        )

    if any(k <= 0 for k in fir_lengths):
        raise ValueError(
            "All FIR filter lengths must be positive."
        )

    if not (0 < learning_rate <= 1):
        raise ValueError(
            "Learning rate must be greater than 0 "
            "and less than or equal to 1."
        )

    if not (0 <= momentum < 1):
        raise ValueError(
            "Momentum must be between 0 and 1."
        )

    # ---------------------------------------
    # Configuration dictionary
    # ---------------------------------------

    config = {
        "num_layers": num_layers,
        "neurons": num_neurons,
        "fir_lengths": fir_lengths,
        "learning_rate": learning_rate,
        "momentum": momentum
    }

    return config


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Covers:
        Python random
        NumPy
        PyTorch CPU
        PyTorch CUDA (if available)
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    """
    Standalone configuration test.

    Run from project root:
        python code/PFG_parameters.py
    """

    root = Path(__file__).resolve().parent.parent

    parameter_file = root / "inputParameters.txt"

    print("\nFIR-MLP PARAMETER TEST")
    print("-" * 45)

    config = load_config(parameter_file)

    set_seed(42)

    print("Configuration loaded successfully!")
    print()

    print("Number of layers:", config["num_layers"])
    print("Neurons:", config["neurons"])
    print("FIR lengths:", config["fir_lengths"])
    print("Learning rate:", config["learning_rate"])
    print("Momentum:", config["momentum"])

    print()
    print("Random seed: 42")

    print("-" * 45)
    print("Parameter validation successful!")
