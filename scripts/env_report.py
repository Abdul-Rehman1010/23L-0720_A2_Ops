"""Task 1.1: print exact versions of the stack and whether TensorFlow sees a GPU."""
import platform
import sys
from importlib import metadata

PACKAGES = ["tensorflow", "keras", "numpy", "pandas", "scikit-learn", "pillow",
            "matplotlib", "mlflow", "dvc", "keras-tuner", "flask", "sqlalchemy"]


def main():
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"Interpreter : {sys.executable}")
    for name in PACKAGES:
        try:
            print(f"{name:<12}: {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"{name:<12}: NOT INSTALLED")
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible to TensorFlow: {gpus if gpus else 'none (CPU only)'}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")


if __name__ == "__main__":
    main()
