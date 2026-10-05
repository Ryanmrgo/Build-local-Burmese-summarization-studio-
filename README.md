# MyanBrief

**MyanBrief** is a local Burmese article summarization studio powered by a fine-tuned mT5 sequence-to-sequence model.

The app uses Gradio for the interface and PyTorch plus Hugging Face Transformers for inference. Articles are processed locally and are not sent to a hosted API.

## Features

- Burmese syllable segmentation matching the training preprocessing
- `summarize:` task prefix
- Local mT5 checkpoint loading
- CUDA when available, CPU fallback otherwise
- Burmese article input and summary output
- Example article loader
- Adjustable generation settings
- Responsive Gradio interface
- Clear local model-file errors

## Repository layout

```text
app.py          Command-line entrypoint
summarizer.py   Model loading, preprocessing, and generation
ui.py           Gradio layout, styling, and event wiring
requirements.txt
README.md
.gitignore
```

The local checkpoint and training artifacts are intentionally excluded from Git by `.gitignore` because `model.safetensors` is large and training state is not needed to run the interface.

## Requirements

- Python 3.10 or newer
- A local exported Hugging Face checkpoint
- Optional CUDA-enabled PyTorch installation for GPU inference

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Add the model checkpoint

Place the exported checkpoint in the repository root, or store it in another local directory. The checkpoint should contain:

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

The included `.gitignore` keeps these files out of commits. For a different local checkpoint path, use `--model-dir` or `BURMESE_MODEL_DIR`:

```powershell
python app.py --model-dir C:\path\to\burmese-checkpoint
```

## Run

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

Use another local port when needed:

```powershell
python app.py --port 7861
```

The tokenizer and model are loaded once when the first summary is generated. CUDA is selected automatically when available; otherwise inference runs on CPU.

## Inference behavior

1. Normalize Burmese whitespace.
2. Apply the training-time Burmese syllable segmentation.
3. Add the `summarize:` prefix.
4. Tokenize with a 512-token input limit.
5. Generate with configurable beam search settings.
6. Decode the summary and remove segmentation spaces.

Generated text is a draft and should be reviewed for factual accuracy, grammar, names, dates, and missing details before publication or classroom use.

## GitHub publishing

After reviewing the files and copying the local checkpoint separately, create the repository with:

```powershell
git init
git add app.py summarizer.py ui.py requirements.txt README.md .gitignore
git commit -m "Build local Burmese summarization studio"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Do not use `git add .` until you confirm that the checkpoint and training artifacts are ignored. GitHub has file-size limits, and the model weights should be distributed separately from the source repository.

## Limitations

- Long articles are truncated to the 512-token model input limit.
- CPU inference can be slow for large inputs.
- Summaries may omit details or contain errors.
- The interface does not retrain or modify the model.
