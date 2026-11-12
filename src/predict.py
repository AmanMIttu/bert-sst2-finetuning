"""Local inference for a saved BERT SST-2 sequence classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

LABELS = ("Negative", "Positive")
DEFAULT_MODEL_DIR = Path("models/best_model")


def _validate_texts(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        values = [texts]
    elif isinstance(texts, Sequence) and not isinstance(texts, (bytes, bytearray)):
        values = list(texts)
    else:
        raise TypeError("text input must be a string or a sequence of strings")
    if not values:
        raise ValueError("provide at least one non-empty text")
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"text at index {index} must be a string")
        if not value.strip():
            raise ValueError(f"text at index {index} is empty; provide non-empty text")
    return values


def load_model(model_dir: str | Path = DEFAULT_MODEL_DIR, *, device: str | None = None):
    """Load a saved model/tokenizer pair from disk and put the model in eval mode."""
    model_path = Path(model_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {model_path}. "
            "Run the training notebook and save the model to this directory first."
        )
    required_groups = [
        ("config.json",),
        ("model.safetensors", "pytorch_model.bin"),
        ("tokenizer.json", "vocab.txt"),
    ]
    missing = [" or ".join(group) for group in required_groups
               if not any((model_path / name).is_file() for name in group)]
    if missing:
        raise FileNotFoundError(
            f"Incomplete fine-tuned model in {model_path}; missing required file(s): "
            + "; ".join(missing)
            + ". Run the training notebook and save both model and tokenizer files."
        )
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install project dependencies with `pip install -r requirements.txt` to load a model.") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"Could not load a fine-tuned model from {model_path}; check its config, weights, and tokenizer files."
        ) from exc
    target_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()
    return model, tokenizer


def predict_sentiment(
    texts: str | Sequence[str],
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    *,
    max_length: int = 128,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Predict sentiment and class probabilities for one or more texts.

    Inputs longer than ``max_length`` wordpieces are truncated by the tokenizer.
    Inject loaded components for repeated predictions and tests.
    """
    values = _validate_texts(texts)
    if not isinstance(max_length, int) or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_dir, device=device)

    import torch
    target_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    model.eval()
    encoded = tokenizer(values, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(target_device) if hasattr(value, "to") else value for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    logits = output.logits if hasattr(output, "logits") else output[0]
    logits = logits.detach().float().cpu().numpy()

    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape != (len(values), 2):
        raise ValueError(f"Expected model logits with shape ({len(values)}, 2), got {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    results = []
    for text, probs in zip(values, probabilities):
        class_id = int(np.argmax(probs))
        results.append({
            "text": text,
            "label": LABELS[class_id],
            "probabilities": {"Negative": float(probs[0]), "Positive": float(probs[1])},
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict SST-2 sentiment with a locally saved fine-tuned BERT model.")
    parser.add_argument("texts", nargs="+", help="one or more input sentences")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="directory containing the saved model and tokenizer")
    parser.add_argument("--max-length", type=int, default=128, help="maximum tokenized length; longer inputs are truncated")
    args = parser.parse_args()
    import json
    print(json.dumps(predict_sentiment(args.texts, args.model_dir, max_length=args.max_length), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
