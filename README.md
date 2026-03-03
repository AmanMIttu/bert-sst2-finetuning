BERT Fine-Tuning on SST-2
Task

Binary sentiment classification (GLUE SST-2).

Model

BERT-base-uncased (transformers 5.2.0).

Setup

Python 3.12

PyTorch (CUDA enabled)

GPU: (fill after training finishes)

Training

Subset: 15k samples

Batch size: 32

Epochs: 2

Learning rate: 2e-5

Weight decay: 0.01

Evaluation

Validation accuracy and confusion matrix analysis included.

Results

Final validation accuracy: 0.9094
Validation loss: 0.2917
Best checkpoint: ./results/checkpoint-938

Compute

Training performed on NVIDIA Tesla T4 (15GB VRAM, CUDA 13.0).
2 epochs, 938 steps (~48 minutes runtime).

Reproducibility

Install dependencies:

pip install -r requirements.txt

Run the notebook to reproduce training and evaluation.



