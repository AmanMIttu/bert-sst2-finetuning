# BERT Fine-Tuning on GLUE SST-2

A reproducible BERT sentiment-classification project with an educational training notebook, saved local model, inference CLI, pytest coverage, and an optional Streamlit UI. The original Colab notebook remains preserved as historical experiment evidence.

## Verified training run

The clean notebook was executed on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GiB VRAM), using Python 3.13.17, PyTorch 2.15.0+cu132, CUDA 13.2, Transformers 5.22.2, and Datasets 4.8.5. The run used FP16 at batch size 8 and completed without an out-of-memory fallback.

| Validation metric | Result |
|---|---:|
| Accuracy | 91.74% |
| Positive-class precision | 90.61% |
| Positive-class recall | 93.47% |
| Positive-class F1 | 92.02% |
| Cross-entropy loss | 0.3789 |

Confusion matrix (true labels as rows, predictions as columns; label order Negative, Positive):

```text
[[385, 43],
 [ 29, 415]]
```

Training took 6m 37s wall time, with peak allocated GPU memory of 1.64 GiB. Validation loss rose from 0.3054 after epoch 1 to 0.3789 after epoch 2, while validation F1 rose from 0.9087 to 0.9202. This pattern is a reason to compare seeds or add early stopping in future work; the current checkpoint is selected by validation F1 as configured.

![Training and validation loss curves](assets/loss_curves.png)

![Validation confusion matrix](assets/confusion_matrix.png)

### Historical result versus this run

The previous README recorded 90.94% validation accuracy, 0.2917 loss, and `checkpoint-938` for a separate 15k/2-epoch run with batch size 32 on a Tesla T4. The original notebook contains an earlier untrained evaluation (50.92% accuracy, 0.6914 loss), but no saved output for the final training cell; its checkpoint and trainer log are absent from the clone. Therefore those old values remain documented historical claims, not independently verified results. The results above come from the newly executed run and its saved validation artifacts. Neither run reports SST-2 test metrics.

## What the project does

- Fine-tunes `bert-base-uncased` for binary movie-review sentiment.
- Uses the first 15,000 rows of the official training split and keeps official validation data separate.
- Selects a checkpoint by validation F1, reports binary positive-class precision/recall/F1, accuracy, loss, a confusion matrix, a classification report, and actual validation errors.
- Saves model/tokenizer files for repeatable CPU or CUDA inference.

## Why BERT?

BERT is a bidirectional Transformer encoder pretrained to build contextual token representations. WordPiece tokenization turns text into input IDs; attention masks indicate real tokens versus padding. Self-attention lets each token use context from the rest of the sentence. Fine-tuning updates pretrained encoder weights together with a two-class head over the `[CLS]` representation. Cross-entropy measures the difference between class logits and labels; AdamW optimizes the model with a linear schedule and 10% warmup.

## Dataset

The notebook loads `nyu-mll/glue`, configuration `sst2`, with Hugging Face Datasets. SST-2 has English movie-review sentences labeled `0` (negative) or `1` (positive). The 15,000-example training subset is selected from the official train split. The official validation split has 872 examples and is used for checkpoint selection and all reported metrics. Test labels are hidden, so no test metric is computed. Sources: [GLUE](https://gluebenchmark.com/) and [Hugging Face GLUE dataset card](https://huggingface.co/datasets/nyu-mll/glue).

## Repository structure

```text
.
├── app.py
├── assets/
│   ├── confusion_matrix.png
│   └── loss_curves.png
├── bert_text_classification_sst2.ipynb  # original, preserved
├── notebooks/
│   └── bert_sst2_finetuning.ipynb       # executed training workflow
├── src/
│   ├── __init__.py
│   └── predict.py
├── tests/
│   ├── test_app.py
│   └── test_predict.py
├── requirements.txt
├── requirements-lock.txt                # exact Windows/Python 3.13 CPU environment
└── README.md
```

The fine-tuned model is **not included in Git**. Model/checkpoint directories are ignored, and there is no verified fine-tuned checkpoint download URL. The current working copy has a local model at `models/best_model/`, but a fresh clone will not have it. Inference defaults to the exact directory `models/best_model/` (`src/predict.py: DEFAULT_MODEL_DIR`).

## Setup

Use Python 3.13 on Windows (the saved training run used 3.13.17). From the repository root, create and install the exact CPU inference/test dependency snapshot:

```powershell
python --version  # confirm Python 3.13
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip check
```

`requirements-lock.txt` pins the verified Python 3.13 Windows environment, including PyTorch 2.15.0 CPU, Transformers 5.22.2, and Datasets 4.8.5. It supports CPU inference and all automated tests. `requirements.txt` keeps the supported version ranges for general environments. Installing dependencies alone does not provide the fine-tuned weights.

The verified training run used PyTorch 2.15.0+cu132 with CUDA 13.2 on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GiB VRAM). For CUDA training, create a separate environment and install the matching CUDA-enabled PyTorch build using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install `requirements.txt`. Do not install the CPU lock into that GPU environment. The notebook checks CUDA and stops instead of silently running full training on CPU.

The public dataset and pretrained weights download on first use, so initial training needs internet access. No Hugging Face token is required for these public files.

## Training

Open and run `notebooks/bert_sst2_finetuning.ipynb` top to bottom. The executed notebook records actual outputs and creates:

- `models/best_model/` — selected model and tokenizer.
- `results/run_config.json` — actual run settings and runtime/hardware metadata.
- `results/validation_metrics.json` — validation metrics, confusion matrix, and classification report.
- `results/trainer_log_history.json`, `results/loss_curves.png`, and `results/confusion_matrix.png`.

The baseline run used 15,000 training examples, max length 128, 2 epochs, batch size 8, evaluation batch size 8, FP16, learning rate `2e-5`, weight decay `0.01`, gradient accumulation 1, linear decay, 375 warmup steps, seed 42, and gradient clipping at 1.0. Checkpoint selection uses validation F1. If CUDA OOM occurs, the notebook clears memory and retries with batch size 4, then batch 4 with gradient checkpointing. It does not silently switch to CPU.

The notebook calls `load_dataset("nyu-mll/glue", "sst2")`, downloads public GLUE SST-2 data and pretrained `bert-base-uncased` weights, then saves the fine-tuned model and tokenizer to the fixed `models/best_model/` directory. Trainer checkpoints and run records are written under `results/baseline_15k/`. Running the notebook again may replace files at those paths. The notebook requires a CUDA-enabled PyTorch environment and a compatible GPU; it does not provide a CPU-training fallback. This notebook is the repository's procedure for generating a new fine-tuned model. It does not reproduce byte-identical weights from the ignored local model associated with the documented metrics, and no separate fine-tuned model download procedure is provided. A fresh clone therefore needs a compatible GPU training run (or a separately obtained compatible model directory) before inference. For local Jupyter, install Jupyter separately with `python -m pip install jupyter` if it is not already available, then run:

```powershell
jupyter notebook notebooks/bert_sst2_finetuning.ipynb
```

## Inference

After cloning, create the CUDA training environment described above and run `notebooks/bert_sst2_finetuning.ipynb` from top to bottom. Confirm that the notebook's final save cell created `models/best_model/` with `config.json`, model weights (`model.safetensors` or `pytorch_model.bin`), and tokenizer files. Then activate an inference environment with the project dependencies (the CPU `requirements-lock.txt` is sufficient for inference), return to the repository root, and run:

```powershell
python -m src.predict --model-dir models/best_model "A delightful, moving film." "A dull and predictable movie."
```

The command returns the text, `Positive`/`Negative` label, and both class probabilities. Long texts are truncated at 128 WordPieces. Empty or invalid inputs and missing/incomplete model files produce actionable errors. To call inference in Python:

```python
from src.predict import load_model, predict_sentiment

model, tokenizer = load_model("models/best_model")  # load once and reuse
predictions = predict_sentiment(
    ["A delightful film."], model=model, tokenizer=tokenizer
)
print(predictions[0])
```

## Optional Streamlit app

Install the optional UI dependency and start the app from the repository root:

```powershell
python -m pip install streamlit
streamlit run app.py
```

The model is cached as a Streamlit resource so it is not loaded again on every rerun. The interface includes sample positive, negative, and mixed reviews, displays both class probabilities, and explains that predictions are estimates that can be wrong.

## Tests and checks

```powershell
python -m pytest -q
python -m pip check
```

Tests use mocked model components, so they do not retrain or download model weights. The app's empty-input handling is tested with a mocked Streamlit interface.

## Limitations and next steps

- Validation results are from one seed; they are not a guarantee of performance on other domains or future examples.
- Validation loss increased in epoch 2 even though F1 improved; early stopping or multiple-seed comparisons may be useful.
- SST-2 contains short English movie-review sentences. Sarcasm, domain shift, other languages, and long documents may reduce performance.
- Softmax probabilities are model estimates, not calibrated confidence intervals.
- The SST-2 test labels are hidden and were not used for model selection or evaluation.

## References

- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
- Wang et al., [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://aclanthology.org/W18-5446/).
- Hugging Face [Transformers](https://huggingface.co/docs/transformers), [Datasets](https://huggingface.co/docs/datasets), and [GLUE dataset card](https://huggingface.co/datasets/nyu-mll/glue).
