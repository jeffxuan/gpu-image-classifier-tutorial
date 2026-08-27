# Local Dog Breed Classifier

This folder contains a Gradio interface for running the fine-tuned ResNet-34 dog-breed classifier locally.

## Required files

Place these files beside `app.py`:

- `resnet34_finetune_best.pt` - trained model checkpoint
- `classes.csv` - the 120 class names, one per row

Model checkpoints and datasets are intentionally excluded from Git because they are large. The original training and GPU inference remain on the remote machine.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Then open `http://127.0.0.1:7860`.
