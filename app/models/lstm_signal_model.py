import json
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from app.config import config

logger = logging.getLogger(__name__)


class StockLSTMModel:
    """Keras LSTM classifier for stock BUY/HOLD/SELL prediction from return sequences."""

    SIGNAL_CLASSES = ["SELL", "HOLD", "BUY"]

    def __init__(
        self,
        sequence_length: int = 60,
        input_features: int = 1,
        lstm_units: int = 64,
        dropout: float = 0.2,
        dense_units: int = 32,
        model_path: Optional[str] = None,
    ):
        self.sequence_length = sequence_length
        self.input_features = input_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.model_path = Path(model_path) if model_path else None
        self.model = None

    def _load_tensorflow(self):
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorFlow is required for stock LSTM training and inference. "
                "Install it with `pip install tensorflow-cpu`."
            ) from exc
        return tf

    def build_model(self):
        tf = self._load_tensorflow()

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.sequence_length, self.input_features)),
                tf.keras.layers.LSTM(self.lstm_units),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(self.dense_units, activation="relu"),
                tf.keras.layers.Dense(len(self.SIGNAL_CLASSES), activation="softmax"),
            ]
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        return model

    def ensure_loaded(self):
        if self.model is not None:
            return
        if self.model_path and self.model_path.exists():
            self.load(str(self.model_path))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_path: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        tf = self._load_tensorflow()
        save_path = save_path or (str(self.model_path) if self.model_path else "models/stock_lstm.h5")
        artifact_path = Path(artifacts_dir or Path(save_path).with_suffix(""))
        artifact_path.mkdir(parents=True, exist_ok=True)

        self.sequence_length = int(X_train.shape[1])
        self.input_features = int(X_train.shape[2])
        self.model = self.build_model()
        self.model.optimizer.learning_rate = learning_rate

        y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=len(self.SIGNAL_CLASSES))
        y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes=len(self.SIGNAL_CLASSES))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=0,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train_encoded,
            validation_data=(X_val, y_val_encoded),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        self.load(save_path)
        probabilities = self.model.predict(X_val, verbose=0)
        predictions = probabilities.argmax(axis=1)
        evaluation = self.model.evaluate(X_val, y_val_encoded, verbose=0)

        cm = confusion_matrix(y_val, predictions, labels=[0, 1, 2])
        self._save_training_artifacts(history.history, cm, artifact_path)

        summary = {
            "best_validation_accuracy": round(float(max(history.history.get("val_accuracy", [0.0])) * 100), 2),
            "final_validation_accuracy": round(float(evaluation[1] * 100), 2),
            "final_validation_loss": round(float(evaluation[0]), 4),
            "epochs_ran": len(history.history.get("loss", [])),
            "confusion_matrix": cm.tolist(),
            "artifacts_dir": str(artifact_path),
            "history_path": str(artifact_path / "history.json"),
            "confusion_matrix_path": str(artifact_path / "confusion_matrix.png"),
            "training_curve_path": str(artifact_path / "training_curves.png"),
        }

        with open(artifact_path / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        return summary

    def predict(self, feature_sequence: np.ndarray) -> dict:
        self.ensure_loaded()
        if self.model is None:
            raise RuntimeError("Stock LSTM model is not available. Train or load a .h5 model first.")

        feature_sequence = np.asarray(feature_sequence, dtype=np.float32)
        if feature_sequence.shape != (self.sequence_length, self.input_features):
            raise ValueError(
                f"Expected input shape {(self.sequence_length, self.input_features)}, "
                f"received {feature_sequence.shape}."
            )

        probabilities = self.model.predict(feature_sequence[np.newaxis, ...], verbose=0)[0]
        predicted_idx = int(np.argmax(probabilities))

        return {
            "signal": self.SIGNAL_CLASSES[predicted_idx],
            "confidence": float(probabilities[predicted_idx]),
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(self.SIGNAL_CLASSES)
            },
        }

    def load(self, path: str):
        tf = self._load_tensorflow()
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path
        logger.info("Loaded stock LSTM model from %s", model_path)

    def _save_training_artifacts(self, history: dict, cm: np.ndarray, artifact_path: Path):
        with open(artifact_path / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        epochs = range(1, len(history.get("loss", [])) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, history.get("loss", []), label="Train Loss")
        axes[0].plot(epochs, history.get("val_loss", []), label="Val Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, history.get("accuracy", []), label="Train Accuracy")
        axes[1].plot(epochs, history.get("val_accuracy", []), label="Val Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(artifact_path / "training_curves.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(self.SIGNAL_CLASSES)))
        ax.set_yticks(range(len(self.SIGNAL_CLASSES)))
        ax.set_xticklabels(self.SIGNAL_CLASSES)
        ax.set_yticklabels(self.SIGNAL_CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Validation Confusion Matrix")
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, int(cm[row, col]), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        plt.tight_layout()
        plt.savefig(artifact_path / "confusion_matrix.png")
        plt.close(fig)


HybridSignalModel = StockLSTMModel


class _NumpySequentialStockModel:
    """Small NumPy inference engine for the attached Keras stock models."""

    def __init__(self, input_shape: tuple[int, int], architecture: list[dict], weights: dict[str, list[np.ndarray]]):
        self.input_shape = tuple(input_shape)
        self.architecture = architecture
        self.weights = weights

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        activations = np.asarray(batch, dtype=np.float32)
        if activations.ndim == 2:
            activations = activations[np.newaxis, ...]
        if activations.ndim != 3:
            raise ValueError(f"Expected a 3D batch tensor, received shape {activations.shape}.")
        if tuple(activations.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, received {tuple(activations.shape[1:])}.")

        for index, layer in enumerate(self.architecture):
            layer_type = layer["type"]
            if layer_type == "lstm":
                activations = self._run_lstm(
                    activations,
                    self.weights[f"lstm_{index}"],
                    return_sequences=layer["return_sequences"],
                )
            elif layer_type == "dropout":
                continue
            elif layer_type == "dense":
                activations = self._run_dense(
                    activations,
                    self.weights[f"dense_{index}"],
                    activation=layer["activation"],
                )
            else:
                raise ValueError(f"Unsupported layer type '{layer_type}' in attached stock model.")
        return activations.astype(np.float32)

    @staticmethod
    def _run_lstm(inputs: np.ndarray, weights: list[np.ndarray], return_sequences: bool) -> np.ndarray:
        kernel, recurrent_kernel, bias = [np.asarray(weight, dtype=np.float32) for weight in weights]
        batch_size, timesteps, _ = inputs.shape
        units = recurrent_kernel.shape[0]

        h_state = np.zeros((batch_size, units), dtype=np.float32)
        c_state = np.zeros((batch_size, units), dtype=np.float32)
        outputs = []

        for step in range(timesteps):
            x_t = inputs[:, step, :]
            z = x_t @ kernel + h_state @ recurrent_kernel + bias
            i_gate, f_gate, c_gate, o_gate = np.split(z, 4, axis=1)
            i_gate = _NumpySequentialStockModel._sigmoid(i_gate)
            f_gate = _NumpySequentialStockModel._sigmoid(f_gate)
            c_gate = np.tanh(c_gate)
            o_gate = _NumpySequentialStockModel._sigmoid(o_gate)

            c_state = f_gate * c_state + i_gate * c_gate
            h_state = o_gate * np.tanh(c_state)
            outputs.append(h_state.copy())

        if return_sequences:
            return np.stack(outputs, axis=1).astype(np.float32)
        return h_state.astype(np.float32)

    @staticmethod
    def _run_dense(inputs: np.ndarray, weights: list[np.ndarray], activation: str) -> np.ndarray:
        kernel, bias = [np.asarray(weight, dtype=np.float32) for weight in weights]
        outputs = inputs @ kernel + bias

        if activation == "relu":
            return np.maximum(outputs, 0.0).astype(np.float32)
        if activation == "softmax":
            shifted = outputs - outputs.max(axis=-1, keepdims=True)
            exps = np.exp(shifted)
            return (exps / exps.sum(axis=-1, keepdims=True)).astype(np.float32)
        if activation in {"linear", None}:
            return outputs.astype(np.float32)
        raise ValueError(f"Unsupported Dense activation '{activation}' in attached stock model.")

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


class AttachedStockModelRouter:
    """Routes a selected stock symbol to its attached `.keras` model file."""

    def __init__(
        self,
        models_dir: str | Path = None,
        pattern: str = None,
        fallback_model_path: str | None = None,
    ):
        self.models_dir = Path(models_dir or config.ATTACHED_MODEL_DIR)
        self.pattern = pattern or config.ATTACHED_MODEL_PATTERN
        self.fallback_model_path = Path(fallback_model_path) if fallback_model_path else None
        self._cache_dir = Path(config.DATA_DIR) / "attached-model-cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, _NumpySequentialStockModel] = {}
        self._model_specs: dict[str, dict] = {}
        self._registry = self._discover_models()

    def supported_symbols(self) -> list[str]:
        return sorted(self._registry.keys())

    def has_symbol(self, symbol: str) -> bool:
        return symbol.strip().upper() in self._registry

    def model_path_for_symbol(self, symbol: str) -> str | None:
        model_path = self._registry.get(symbol.strip().upper())
        return str(model_path) if model_path else None

    def model_details(self) -> list[dict]:
        details = []
        for symbol in self.supported_symbols():
            spec = self._model_specs.get(symbol, {})
            details.append(
                {
                    "symbol": symbol,
                    "model_path": self.model_path_for_symbol(symbol),
                    "input_shape": spec.get("input_shape"),
                    "sequence_length": spec.get("input_shape", [None, None])[0],
                    "input_features": spec.get("input_shape", [None, None])[1],
                    "feature_pipeline": {
                        "feature_column": config.LSTM_FEATURE_COLUMN,
                        "sequence_normalization": config.LSTM_SEQUENCE_NORMALIZATION,
                    },
                }
            )
        return details

    def predict(self, symbol: str, feature_sequence: np.ndarray) -> dict:
        normalized_symbol = symbol.strip().upper()
        model = self._get_model(normalized_symbol)
        feature_sequence = np.asarray(feature_sequence, dtype=np.float32)
        probabilities = model.predict(feature_sequence[np.newaxis, ...], verbose=0)[0]
        signal_classes = ["SELL", "HOLD", "BUY"]
        predicted_idx = int(np.argmax(probabilities))
        return {
            "signal": signal_classes[predicted_idx],
            "confidence": float(probabilities[predicted_idx]),
            "probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(signal_classes)
            },
            "model_path": self.model_path_for_symbol(normalized_symbol),
        }

    def _discover_models(self) -> dict[str, Path]:
        registry = {}
        for model_path in self.models_dir.glob(self.pattern):
            symbol = model_path.name.replace("_lstm_model.keras", "").upper()
            registry[symbol] = model_path.resolve()
            self._model_specs[symbol] = self._read_model_spec(model_path.resolve())
        logger.info("Discovered attached stock models for symbols: %s", sorted(registry))
        return registry

    def _get_model(self, symbol: str):
        if symbol in self._models:
            return self._models[symbol]

        model_path = self._registry.get(symbol)
        if model_path is None:
            raise FileNotFoundError(
                f"No attached stock model was found for {symbol}. Supported symbols: {', '.join(self.supported_symbols())}"
            )

        model = self._load_attached_model(model_path)
        self._models[symbol] = model
        return model

    def _load_attached_model(self, model_path: Path):
        with zipfile.ZipFile(model_path) as archive:
            config_payload = json.loads(archive.read("config.json"))
            weights_bytes = archive.read("model.weights.h5")

        model = _NumpySequentialStockModel(
            input_shape=self._extract_input_shape(config_payload),
            architecture=self._extract_architecture(config_payload),
            weights=self._extract_weights(weights_bytes),
        )
        logger.info("Loaded attached stock model from %s", model_path)
        return model

    def _read_model_spec(self, model_path: Path) -> dict:
        with zipfile.ZipFile(model_path) as archive:
            config_payload = json.loads(archive.read("config.json"))
            metadata_payload = json.loads(archive.read("metadata.json"))

        return {
            "model_path": str(model_path),
            "input_shape": list(self._extract_input_shape(config_payload)),
            "keras_version": metadata_payload.get("keras_version"),
            "date_saved": metadata_payload.get("date_saved"),
        }

    def _extract_input_shape(self, payload: dict) -> tuple[int, int]:
        layers = payload.get("config", {}).get("layers", [])
        input_config = next(layer["config"] for layer in layers if layer["class_name"] == "InputLayer")
        return tuple(input_config["batch_shape"][1:])

    def _extract_architecture(self, payload: dict) -> list[dict]:
        layers = payload.get("config", {}).get("layers", [])
        architecture = []
        for layer in layers:
            class_name = layer["class_name"]
            config_payload = layer["config"]
            if class_name == "InputLayer":
                continue
            if class_name == "LSTM":
                architecture.append(
                    {
                        "type": "lstm",
                        "units": config_payload["units"],
                        "return_sequences": config_payload["return_sequences"],
                    }
                )
            elif class_name == "Dropout":
                architecture.append({"type": "dropout", "rate": config_payload["rate"]})
            elif class_name == "Dense":
                architecture.append(
                    {
                        "type": "dense",
                        "units": config_payload["units"],
                        "activation": config_payload["activation"],
                    }
                )
        return architecture

    def _extract_weights(self, weights_bytes: bytes) -> dict[str, list[np.ndarray]]:
        with h5py.File(io.BytesIO(weights_bytes), "r") as handle:
            return {
                "lstm_0": [
                    handle["layers/lstm/cell/vars/0"][()],
                    handle["layers/lstm/cell/vars/1"][()],
                    handle["layers/lstm/cell/vars/2"][()],
                ],
                "lstm_2": [
                    handle["layers/lstm_1/cell/vars/0"][()],
                    handle["layers/lstm_1/cell/vars/1"][()],
                    handle["layers/lstm_1/cell/vars/2"][()],
                ],
                "dense_4": [
                    handle["layers/dense/vars/0"][()],
                    handle["layers/dense/vars/1"][()],
                ],
                "dense_5": [
                    handle["layers/dense_1/vars/0"][()],
                    handle["layers/dense_1/vars/1"][()],
                ],
            }
