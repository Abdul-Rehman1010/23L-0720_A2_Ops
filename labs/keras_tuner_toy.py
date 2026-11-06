"""Task 2.7 toy lab: Keras Tuner random search over learning rate and number of units (Fashion-MNIST subset).

  python labs/keras_tuner_toy.py             # run the search (about a minute)
  python labs/keras_tuner_toy.py --summary   # print the saved results table (for the screenshot)
"""
import argparse
import sys
from pathlib import Path

if "--summary" in sys.argv:      # print saved results without importing TensorFlow
    print((Path(__file__).resolve().parents[1] / "reports" / "tables" / "2.7_toy_tuner.md").read_text(encoding="utf-8"))
    raise SystemExit

import keras
import keras_tuner as kt
from keras import layers, optimizers
from keras.datasets import fashion_mnist

HERE = Path(__file__).resolve().parent
TABLE = HERE.parent / "reports" / "tables" / "2.7_toy_tuner.md"
SEED, TRIALS = 42, 8


def build(hp):
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    units = hp.Int("units", 32, 256, step=32)
    model = keras.Sequential([layers.Input((28, 28)), layers.Flatten(),
                              layers.Dense(units, activation="relu"), layers.Dense(10, activation="softmax")])
    model.compile(optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    if ap.parse_args().summary:
        print(TABLE.read_text(encoding="utf-8"))
        return
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    keras.utils.set_random_seed(SEED)
    (x_tr, y_tr), (x_va, y_va) = fashion_mnist.load_data()
    x_tr, y_tr, x_va, y_va = x_tr[:10000] / 255.0, y_tr[:10000], x_va[:2000] / 255.0, y_va[:2000]

    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=TRIALS, seed=SEED,
                            directory=str(HERE / "tuner_runs"), project_name="fashion_toy", overwrite=True)
    tuner.search(x_tr, y_tr, validation_data=(x_va, y_va), epochs=3, batch_size=64, verbose=0)

    trials = tuner.oracle.get_best_trials(num_trials=TRIALS)
    rows = ["| rank | trial | learning_rate | units | best val_accuracy |", "|---|---|---|---|---|"]
    for rank, t in enumerate(trials, start=1):
        rows.append(f"| {rank} | {t.trial_id} | {t.hyperparameters.get('learning_rate'):.5f} | "
                    f"{t.hyperparameters.get('units')} | {t.score:.4f} |")
    best = tuner.get_best_hyperparameters(1)[0]
    rows += ["", f"Best settings: learning_rate = {best.get('learning_rate'):.5f}, units = {best.get('units')}"]
    TABLE.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
