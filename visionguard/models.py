"""Tasks 2.2 / 2.3: the baseline CNN and the transfer-learning models.

Pixel scaling is part of each model (input is float 0..255 from the pipeline), so training and
serving only share preprocessing.resize.
  python -m visionguard.models --summary cnn
  python -m visionguard.models --trainable mobilenetv2
"""
import argparse

from keras import Input, Model, applications, layers, metrics, optimizers

from visionguard import config
from visionguard.preprocessing import IMG_SIZE

# backbone -> (constructor, first layer of the last block, input scaling the weights expect)
BACKBONES = {
    "mobilenetv2": (applications.MobileNetV2, "block_16_expand", layers.Rescaling(1 / 127.5, offset=-1)),
    "efficientnetb0": (applications.EfficientNetB0, "block7a_expand_conv", None),  # scales inside
}


def build_cnn(img_size: int = IMG_SIZE, dropout: float = 0.4) -> Model:
    x = inputs = Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(x)
    for i, filters in enumerate((32, 64, 128), start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(momentum=0.9, name=f"bn{i}")(x)  # 0.99 is too slow for ~22 steps/epoch
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return Model(inputs, outputs, name="visionguard_cnn")


def build_transfer(backbone: str = "mobilenetv2", img_size: int = IMG_SIZE, dropout: float = 0.3) -> Model:
    ctor, _, scale = BACKBONES[backbone]
    base = ctor(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base.trainable = False                                   # stage A: frozen backbone
    x = inputs = Input((img_size, img_size, 3), name="image")
    if scale is not None:
        x = layers.Rescaling(scale.scale, offset=scale.offset, name="backbone_scaling")(x)
    x = base(x, training=False)                              # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return Model(inputs, outputs, name=f"visionguard_{backbone}")


def backbone_of(model: Model) -> Model:
    return next(l for l in model.layers if isinstance(l, Model))


def unfreeze(model: Model, backbone: str | None = None, last_n: int | None = None) -> int:
    """Stage B: make the last block (or the last `last_n` layers) of the backbone trainable.
    BatchNorm layers stay frozen. Returns the number of unfrozen backbone layers that have weights."""
    base = backbone_of(model)
    if last_n is not None:
        start = len(base.layers) - last_n
    else:
        names = [l.name for l in base.layers]
        start = names.index(BACKBONES[backbone][1])
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return sum(l.trainable and bool(l.weights) for l in base.layers)


def compile_model(model: Model, lr: float) -> Model:
    model.compile(optimizers.Adam(lr), "binary_crossentropy",
                  metrics=["accuracy", metrics.AUC(name="auc"), metrics.Precision(name="precision"),
                           metrics.Recall(name="recall")])
    return model


def trainable_layers(model: Model) -> list[str]:
    names = [l.name for l in model.layers if l.trainable and l.weights and not isinstance(l, Model)]
    base = next((l for l in model.layers if isinstance(l, Model)), None)
    if base is not None and base.trainable:
        names += [f"{base.name}/{l.name}" for l in base.layers if l.trainable and l.weights]
    return names


def param_counts(model: Model) -> dict:
    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    total = model.count_params()
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def layer_table(model: Model) -> str:
    rows = ["| layer | type | output shape | params |", "|---|---|---|---|"]
    for l in model.layers:
        shape = l.output.shape if hasattr(l, "output") else ""
        rows.append(f"| {l.name} | {type(l).__name__} | {tuple(shape)} | {l.count_params():,} |")
    c = param_counts(model)
    rows.append(f"| **total** | | | **{c['total']:,}** (trainable {c['trainable']:,}) |")
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", choices=["cnn", *BACKBONES])
    ap.add_argument("--trainable", choices=list(BACKBONES))
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = ap.parse_args()
    if args.summary == "cnn":
        m = build_cnn(args.img_size)
        m.summary()
        out = config.TABLES / "2.2_cnn_layers.md"
        out.write_text(layer_table(m) + "\n", encoding="utf-8")
        print("layer table written to", out)
    elif args.summary:
        build_transfer(args.summary, args.img_size).summary()
    if args.trainable:
        m = build_transfer(args.trainable, args.img_size)
        lines = []
        a = trainable_layers(m); c = param_counts(m)
        lines += [f"STAGE A (backbone frozen): {len(a)} trainable layers, {c['trainable']:,} trainable / {c['total']:,} params"]
        lines += [f"  - {n}" for n in a]
        n = unfreeze(m, args.trainable)
        b = trainable_layers(m); c = param_counts(m)
        lines += ["", f"STAGE B (last block unfrozen, BatchNorm kept frozen): {n} backbone layers + head = "
                  f"{len(b)} trainable layers, {c['trainable']:,} trainable / {c['total']:,} params"]
        lines += [f"  - {x}" for x in b]
        print("\n".join(lines))
        (config.TABLES / f"2.3_trainable_{args.trainable}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
