"""
RNN & LSTM Case Study 

Task: Sentiment classification on IMDB reviews (binary: positive/negative).

Why this case study?
- Classic NLP sequence task
- Shows the need for "memory" in sequences
- Great for comparing SimpleRNN vs GRU vs LSTM vs BiLSTM
- Runs fast on CPU with small epochs + smart callbacks

How to run:
1) pip install tensorflow matplotlib numpy
2) python rnn_lstm_case_study.py

Tips:
- If training is slow, reduce MAX_LEN or BATCH_SIZE.
- If accuracy plateaus, increase vocab (MAX_FEATURES) or use BiLSTM.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TF info logs

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1) Reproducibility
# -----------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 2) Core Hyperparameters (tune these)
# -----------------------------
MAX_FEATURES = 12000
MAX_LEN = 120
EMBED_DIM = 96
RNN_UNITS = 96
DROPOUT = 0.25
REC_DROPOUT = 0.0
BATCH_SIZE = 64
EPOCHS = 8
LEARNING_RATE = 2e-3
TARGET_VAL_ACCURACY = 0.86

# Callback stop target (adjust based on your machine & run)
TARGET_VAL_ACCURACY = 0.88  # stop when val_accuracy >= this

# -----------------------------
# 3) Custom callback to stop at target accuracy
# -----------------------------
class StopAtValAccuracy(callbacks.Callback):
    """Stops training when validation accuracy reaches a target."""
    def __init__(self, target=0.88, verbose=1):
        super().__init__()
        self.target = target
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy")
        if val_acc is not None and val_acc >= self.target:
            if self.verbose:
                print(f"\n Reached target val_accuracy={val_acc:.4f} (>= {self.target}). Stopping.")
            self.model.stop_training = True

# -----------------------------
# 4) Load & prepare data
# -----------------------------
def load_data(max_features=MAX_FEATURES, max_len=MAX_LEN):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = pad_sequences(x_train, maxlen=max_len, padding="post", truncating="post")
    x_test  = pad_sequences(x_test,  maxlen=max_len, padding="post", truncating="post")
    return (x_train, y_train), (x_test, y_test)

# -----------------------------
# 5) Model Builder (choose architecture)
# -----------------------------
def build_model(model_type="bilstm",
                max_features=MAX_FEATURES,
                max_len=MAX_LEN,
                embed_dim=EMBED_DIM,
                units=RNN_UNITS,
                dropout=DROPOUT,
                rec_dropout=REC_DROPOUT):
    """
    model_type options:
      - "simplernn" : SimpleRNN baseline
      - "gru"       : GRU
      - "lstm"      : LSTM
      - "bilstm"    : Bidirectional LSTM (default)
    """
    inputs = layers.Input(shape=(max_len,), name="token_ids")
    x = layers.Embedding(input_dim=max_features, output_dim=embed_dim, input_length=max_len, name="embedding")(inputs)

    if model_type == "simplernn":
        x = layers.SimpleRNN(units, dropout=dropout, recurrent_dropout=rec_dropout, name="simplernn")(x)
    elif model_type == "gru":
        x = layers.GRU(units, dropout=dropout, recurrent_dropout=rec_dropout, name="gru")(x)
    elif model_type == "lstm":
        x = layers.LSTM(units, dropout=dropout, recurrent_dropout=rec_dropout, name="lstm")(x)
    elif model_type == "bilstm":
        x = layers.Bidirectional(layers.LSTM(units, dropout=dropout, recurrent_dropout=rec_dropout), name="bilstm")(x)
    else:
        raise ValueError("model_type must be one of: simplernn, gru, lstm, bilstm")

    x = layers.Dense(64, activation="relu", name="dense_relu")(x)
    x = layers.Dropout(0.35, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return models.Model(inputs, outputs, name=f"imdb_{model_type}")

# -----------------------------
# 6) Train & evaluate
# -----------------------------
def plot_history(history, model_type="model"):
    hist = history.history
    epochs_ran = range(1, len(hist.get("loss", [])) + 1)

    plt.figure()
    plt.plot(epochs_ran, hist.get("loss", []), label="train_loss")
    plt.plot(epochs_ran, hist.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({model_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_type}_loss.png", dpi=150)
    plt.show()

    plt.figure()
    plt.plot(epochs_ran, hist.get("accuracy", []), label="train_acc")
    plt.plot(epochs_ran, hist.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve ({model_type})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_type}_accuracy.png", dpi=150)
    plt.show()

def train_model(model_type="bilstm"):
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model(model_type=model_type)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel summary:")
    model.summary()

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(filepath=f"best_{model_type}.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
        StopAtValAccuracy(target=TARGET_VAL_ACCURACY, verbose=1),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=2
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f" Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    plot_history(history, model_type=model_type)
    return model, history, (test_loss, test_acc)

def run_experiment():
    """Quick comparison across architectures."""
    results = {}
    for mtype in ["simplernn", "gru", "lstm", "bilstm"]:
        print("\n" + "=" * 70)
        print(f"Training model: {mtype.upper()}")
        print("=" * 70)
        _, _, (t_loss, t_acc) = train_model(model_type=mtype)
        results[mtype] = {"test_loss": float(t_loss), "test_acc": float(t_acc)}

    print("\nFinal comparison (test accuracy):")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["test_acc"], reverse=True):
        print(f"- {k:9s}: {v['test_acc']:.4f}")

    return results

if __name__ == "__main__":
    RUN_COMPARISON = False  # set True to compare all models
    if RUN_COMPARISON:
        run_experiment()
    else:
        train_model(model_type="bilstm")
