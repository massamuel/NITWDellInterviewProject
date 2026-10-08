"""Train a reproducible 16-class Keras text classifier with held-out evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from backend.text import LABELS, clean_text


def train(args):
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    print("Loading and cleaning dataset...", flush=True)
    df = pd.read_csv(args.data).dropna(subset=["posts", "type"])
    df["text"] = df.posts.map(clean_text)
    df = df[df.text.str.len() > 0].drop_duplicates("text")
    if set(df.type) != set(LABELS):
        raise ValueError("Dataset must contain all 16 MBTI labels")
    x = df.text.to_numpy()
    y = df.type.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    # Test set never participates in vocabulary fitting, training, or early stopping.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2,
        stratify=y, random_state=args.seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2,
        stratify=y_train, random_state=args.seed)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=12000,
        output_mode="tf_idf", standardize=None)
    print("Fitting training-only vocabulary...", flush=True)
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(x_train).batch(64))
    # Precompute features to keep training inexpensive on CPU. Export includes vectorizer.
    def features(texts):
        return np.concatenate([vectorizer(tf.constant(texts[i:i+64])).numpy()
                               for i in range(0, len(texts), 64)])
    classifier = tf.keras.Sequential([
        tf.keras.Input(shape=(len(vectorizer.get_vocabulary()),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(.4),
        tf.keras.layers.Dense(len(LABELS), activation="softmax")])
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0005),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    weights = compute_class_weight("balanced", classes=np.arange(16), y=y_train)
    print("Vectorizing training and validation data...", flush=True)
    history = classifier.fit(features(x_train), y_train, validation_data=(features(x_val), y_val),
        epochs=args.epochs, batch_size=64, class_weight=dict(enumerate(weights)),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                   restore_best_weights=True)], verbose=2)
    inputs = tf.keras.Input(shape=(), dtype=tf.string)
    model = tf.keras.Model(inputs, classifier(vectorizer(inputs)))
    probabilities = model(tf.constant(x_test), training=False).numpy()
    predicted = probabilities.argmax(axis=1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.keras")
    metadata = {"labels": LABELS, "seed": args.seed,
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "model_version": hashlib.sha256((output / "model.keras").read_bytes()).hexdigest()[:12],
        "split_rows": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "epochs_run": len(history.history["loss"]),
        "accuracy": accuracy_score(y_test, predicted),
        "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        "majority_baseline_accuracy": float(np.mean(y_test == np.bincount(y_train).argmax())),
        "classification_report": classification_report(y_test, predicted,
            labels=list(range(16)), target_names=LABELS, output_dict=True, zero_division=0)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in metadata.items() if k != "classification_report"}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mbti_1.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
