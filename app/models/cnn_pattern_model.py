import logging
from pathlib import Path

import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models    
   

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

from app.config import config

logger = logging.getLogger(__name__)


def _safe_model_weights(model_name: str):
    logger.info("Using local/randomly initialized %s backbone weights.", model_name)
    return None


def _maybe_load_pattern_classes_from_dataset_csv(expected_count: int) -> list[str] | None:
    """
    Try to load class names from the Roboflow-style `_classes.csv` header.

    Expected format (first line):
      filename, ClassA, ClassB, ...
    """
    try:
        # Repo root = .../app/models/ -> parents[2] == repo root (`d:\Mv project`)
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            repo_root / "Chart-pattern.v2i.multiclass" / "train" / "_classes.csv",
            repo_root / "Chart-pattern.v2i.multiclass" / "valid" / "_classes.csv",
            repo_root / "Chart-pattern.v2i.multiclass" / "test" / "_classes.csv",
        ]

        for csv_path in candidates:
            if not csv_path.exists():
                continue

            header = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
            if not header:
                continue

            parts = [p.strip() for p in header.split(",") if p.strip()]
            if not parts:
                continue

            # Drop the leading `filename` column if present.
            if parts[0].lower() in {"filename", "file", "image", "image_name"}:
                parts = parts[1:]

            if len(parts) == expected_count:
                logger.info("Loaded %s pattern classes from %s", expected_count, csv_path)
                return parts

    except Exception as exc:
        logger.warning("Failed loading dataset class names from _classes.csv: %s", exc)

    return None


def _default_pattern_classes(count: int) -> list[str]:
    if count == 10:
        return [
            "Double Bottom",
            "Double Top",
            "Head and Shoulders",
            "Inverse Head and Shoulders",
            "Triangle Ascending",
            "Triangle Descending",
            "Flag",
            "Wedge",
            "Cup and Handle",
            "Consolidation",
        ]
    return [f"Pattern Class {index + 1}" for index in range(count)]


class ImprovedPatternDetectionModel(nn.Module):
    """Legacy PyTorch pattern model kept for backward compatibility."""

    PATTERN_CLASSES = _default_pattern_classes(10)

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.num_classes = len(self.PATTERN_CLASSES)
        self.backbone = models.efficientnet_b0(weights=_safe_model_weights("efficientnet_b0"))
        backbone_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.attention = nn.Sequential(
            nn.Linear(backbone_features, backbone_features // 4),
            nn.ReLU(),
            nn.Linear(backbone_features // 4, backbone_features),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, self.num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        combined_features = torch.cat([attended_features, attended_features], dim=1)
        return self.classifier(combined_features)


class PatternDetectionModel:
    """Chart-pattern inference wrapper supporting the attached TensorFlow `.h5` model."""

    PATTERN_CLASSES = _default_pattern_classes(20)

    def __init__(self, model_path: str = None, use_improved_model: bool = True):
        self.model_path = Path(model_path) if model_path else None
        self.use_improved_model = use_improved_model
        self.framework = "numpy-h5"
        self.input_size = (128, 128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_source = "generic"
        self.model = None
        self.bias_detected = False

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Pattern model file not found: {self.model_path}")

        suffix = self.model_path.suffix.lower()
        if suffix in {".h5", ".keras"}:
            self._load_keras_model()
        else:
            self.framework = "pytorch"
            self._load_pytorch_pattern_model()

    def _resolve_pattern_classes(self, class_count: int) -> list[str]:
        configured = config.PATTERN_CLASS_NAMES or []
        if configured and len(configured) == class_count:
            self.label_source = "configured"
            return configured

        dataset_classes = _maybe_load_pattern_classes_from_dataset_csv(class_count)
        if dataset_classes:
            self.label_source = "dataset-csv"
            return dataset_classes

        self.label_source = "generic" if class_count != 10 else "default-10"
        return _default_pattern_classes(class_count)

    def _load_keras_model(self):
        """Load the attached chart-pattern `.h5` model robustly."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to load Keras models. Install with: pip install tensorflow")

        try:
            # First try standard loader (works when Keras versions match).
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            self.framework = "tensorflow"
        except Exception as exc:
            # Fallback: rebuild the known architecture manually and load weights.
            logger.warning("load_model failed (%s). Rebuilding model architecture manually.", exc)
            self.model = self._build_chart_pattern_model_arch()
            self.model.load_weights(self.model_path)
            self.framework = "tensorflow"

        # Model is now loaded.
        input_shape = self.model.input_shape
        self.input_size = (input_shape[1], input_shape[2]) if len(input_shape) >= 3 else (128, 128)
        output_shape = self.model.output_shape
        num_classes = output_shape[-1] if output_shape else 20
        self.PATTERN_CLASSES = self._resolve_pattern_classes(num_classes)

        self._check_model_bias()
        logger.info("Loaded chart pattern model from %s with input size %s", self.model_path, self.input_size)

    def _build_chart_pattern_model_arch(self) -> "tf.keras.Model":
        """
        Manual architecture match for `chart_pattern_tf_model.h5`.

        This avoids Keras deserialization issues (quantization_config, registry problems).
        """
        inputs = tf.keras.Input(shape=(128, 128, 3), name="input_layer_5")
        x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling_5")(inputs)
        x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="valid", name="conv2d_11")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", name="max_pooling2d_11")(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="valid", name="conv2d_12")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", name="max_pooling2d_12")(x)
        x = tf.keras.layers.Flatten(name="flatten_4")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="dense_10")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout_6")(x)
        outputs = tf.keras.layers.Dense(20, activation="softmax", name="dense_11")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="sequential_5")

    def _check_model_bias(self, num_test_samples=50, bias_threshold=0.6):
        """Check if model shows bias towards specific classes."""
        try:
            import numpy as np

            predictions = []
            for _ in range(num_test_samples):
                # Match the real inference range expected by the preprocessing path.
                # The attached Keras chart model already contains a Rescaling(1/255)
                # layer, so app inputs reach the model in raw pixel space.
                test_img = np.random.randint(
                    0,
                    256,
                    size=(1, *self.input_size, 3),
                    dtype=np.uint8,
                ).astype(np.float32)
                pred = self.model.predict(test_img, verbose=0)
                predicted_class = np.argmax(pred[0])
                predictions.append(predicted_class)

            # Check distribution
            unique, counts = np.unique(predictions, return_counts=True)
            max_frequency = max(counts) / num_test_samples

            if max_frequency > bias_threshold:
                dominant_class = unique[np.argmax(counts)]
                logger.warning(
                    "Model bias detected! Class %s predicted %.1f%% of the time. "
                    "Model may need retraining with balanced data.",
                    dominant_class, max_frequency * 100
                )
                # Store bias info for later use
                self.bias_detected = True
                self.dominant_class = dominant_class
                self.bias_confidence = max_frequency
            else:
                logger.info("Model bias check passed - predictions are well-distributed")
                self.bias_detected = False

        except Exception as e:
            logger.warning("Could not check model bias: %s", e)
            self.bias_detected = False

    def _load_pytorch_pattern_model(self):
        self.PATTERN_CLASSES = self._resolve_pattern_classes(10)
        if self.use_improved_model:
            model = ImprovedPatternDetectionModel()
        else:
            model = models.mobilenet_v2(weights=_safe_model_weights("mobilenet_v2"))
            model.classifier[1] = nn.Linear(1280, len(self.PATTERN_CLASSES))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        self.model = model
        self.input_size = (224, 224)
        logger.info("Loaded PyTorch chart pattern model from %s", self.model_path)

    def predict(self, image_batch) -> dict:
        if self.framework == "tensorflow":
            probabilities = self._predict_tensorflow_model(np.asarray(image_batch, dtype=np.float32))[0]
            return self._format_prediction(probabilities)

        with torch.no_grad():
            image_batch = image_batch.to(self.device)
            logits = self.model(image_batch)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return self._format_prediction(probabilities)

    def _predict_tensorflow_model(self, image_batch: np.ndarray) -> np.ndarray:
        """Run prediction using TensorFlow/Keras model."""
        # The attached Keras model already includes a Rescaling(1/255) layer.
        # Feeding pre-divided inputs would normalize twice and collapse outputs.
        image_batch = np.asarray(image_batch, dtype=np.float32)
        predictions = self.model.predict(image_batch, verbose=0)
        return predictions

    def _format_prediction(self, probabilities: np.ndarray) -> dict:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        predicted_idx = int(np.argmax(probabilities))
        predicted_pattern = self.PATTERN_CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        probability_map = {
            pattern: float(probability)
            for pattern, probability in zip(self.PATTERN_CLASSES, probabilities)
        }
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_5 = [
            {
                "pattern": self.PATTERN_CLASSES[index],
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ]

        return {
            "pattern": predicted_pattern,
            "pattern_confidence": confidence,
            "probabilities": probability_map,
            "top_5": top_5,
            "framework": self.framework,
            "label_source": self.label_source,
        }

    # Heuristic fallback removed by request: always use the loaded model output.
