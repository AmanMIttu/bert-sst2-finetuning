import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from src.predict import load_model, predict_sentiment


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def to(self, _device):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        assert all(isinstance(text, str) for text in texts)
        return {"input_ids": FakeTensor([[1, 2] for _ in texts]), "attention_mask": FakeTensor([[1, 1] for _ in texts])}


class FakeModel:
    def __init__(self):
        self.is_eval = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self

    def __call__(self, **inputs):
        count = len(inputs["input_ids"].value)
        logits = [[2.0, 1.0] if index == 0 else [0.1, 2.0] for index in range(count)]
        return SimpleNamespace(logits=FakeTensor(logits))


@pytest.fixture
def components(monkeypatch):
    class TorchStub(types.ModuleType):
        @staticmethod
        def no_grad():
            from contextlib import nullcontext
            return nullcontext()

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def cuda_is_available():
            return False

    torch = TorchStub("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return FakeModel(), FakeTokenizer()


def test_prediction_output_labels_probabilities_and_eval_mode(components):
    model, tokenizer = components
    results = predict_sentiment(["not good", "great"], model=model, tokenizer=tokenizer)
    assert [row["label"] for row in results] == ["Negative", "Positive"]
    assert set(results[0]) == {"text", "label", "probabilities"}
    for row in results:
        assert set(row["probabilities"]) == {"Negative", "Positive"}
        assert all(0.0 <= p <= 1.0 for p in row["probabilities"].values())
        assert sum(row["probabilities"].values()) == pytest.approx(1.0)
    assert model.is_eval


def test_single_string_is_accepted(components):
    model, tokenizer = components
    assert len(predict_sentiment("A lovely film", model=model, tokenizer=tokenizer)) == 1


@pytest.mark.parametrize("text", ["", "   ", ["fine", ""]])
def test_empty_text_is_rejected(text):
    with pytest.raises(ValueError, match="empty"):
        predict_sentiment(text, model=FakeModel(), tokenizer=FakeTokenizer())


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        predict_sentiment([], model=FakeModel(), tokenizer=FakeTokenizer())


def test_non_string_is_rejected():
    with pytest.raises(TypeError, match="must be a string"):
        predict_sentiment(["okay", 3], model=FakeModel(), tokenizer=FakeTokenizer())


def test_missing_model_directory_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="training notebook"):
        predict_sentiment("A movie", model_dir=tmp_path / "missing")


def test_long_text_uses_truncation_and_max_length(components):
    model, tokenizer = components
    long_text = "word " * 5000
    predict_sentiment(long_text, model=model, tokenizer=tokenizer, max_length=64)
    assert tokenizer.kwargs["truncation"] is True
    assert tokenizer.kwargs["max_length"] == 64


def test_incomplete_model_directory_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing required file"):
        predict_sentiment("A movie", model_dir=tmp_path)



def test_model_loading_sets_eval_mode(tmp_path, monkeypatch):
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (tmp_path / name).write_text("stub")

    loaded = FakeModel()
    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return loaded
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeTokenizer()
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForSequenceClassification = FakeAutoModel
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    # Keep this unit test independent of the large PyTorch runtime. The loader
    # only needs device selection and the model's ``to``/``eval`` methods here.
    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    model, tokenizer = load_model(tmp_path, device="cpu")
    assert model is loaded
    assert tokenizer is not None
    assert model.is_eval
    assert str(model.device) == "cpu"
