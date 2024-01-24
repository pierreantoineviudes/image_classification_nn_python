import numpy as np
import pickle
from typing import Tuple


def read_cifar_batch(BATCH_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a CIFAR batch from the specified path

    Args:
        BATCH_PATH (str): The path to the CIFAR batch file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: a tuple containing the data and labels.
    """
    with open(BATCH_PATH, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        data = d[b"data"].astype(np.float32)
        labels = np.array(d[b"labels"]).astype(np.int64)
    return data, labels


def read_cifar():
    pass
