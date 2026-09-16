import numpy as np
from typing import Dict, List, Tuple, Union


class TemperatureScaler:
    """
    Post-hoc probability calibration via Temperature Scaling.
    Optimizes a single scalar parameter T > 0 on validation logits using NLL.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-3, float(temperature))

    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.scale_logits(logits)
        # Numerically stable softmax
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, epochs: int = 100):
        """Fit optimal temperature using gradient descent on Cross-Entropy loss."""
        t = self.temperature
        for _ in range(epochs):
            # Compute softmax probabilities
            scaled = logits / t
            exp_s = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            
            # Cross entropy gradient w.r.t temperature T
            # dL/dT = (1/T^2) * sum_i (z_i * (p_i - y_i))
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(len(labels)), labels] = 1.0
            
            grad = np.sum((probs - one_hot) * logits) / (t ** 2 * len(labels))
            t = max(1e-2, t - lr * grad)
        self.temperature = float(t)
        return self


class PlattScaler:
    """Logistic regression scaling for binary classification calibration."""
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = a
        self.b = b

    def scale_prob(self, score: float) -> float:
        z = self.a * score + self.b
        return float(1.0 / (1.0 + np.exp(-z)))


class ExpectedCalibrationError:
    """
    Computes ECE (Expected Calibration Error) and MCE (Maximum Calibration Error)
    to mathematically measure calibration quality.
    """
    @staticmethod
    def calculate(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> Dict[str, float]:
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        accuracies = predictions == labels

        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        mce = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += diff * prop_in_bin
                mce = max(mce, diff)

        return {"ece": float(ece), "mce": float(mce)}
