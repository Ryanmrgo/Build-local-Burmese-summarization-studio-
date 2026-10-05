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

## Model reference

The following sections are formatted like presentation slides and can be reused in a project presentation.

### Slide 1: Model overview

- **Model:** Fine-tuned mT5-small
- **Task:** Burmese article summarization
- **Model type:** Text-to-text encoder-decoder Transformer
- **Framework:** PyTorch and Hugging Face Transformers
- **Runtime:** Local inference only
- **Output language:** Burmese

The model receives a Burmese article and generates a shorter Burmese draft that preserves the main idea of the source text.

### Slide 2: Transformer architecture

mT5-small uses an encoder-decoder architecture:

```text
Burmese article
	|
	v
Syllable segmentation + "summarize:" prefix
	|
	v
Tokenizer and encoder
	|
	v
Decoder with beam search
	|
	v
Burmese summary
```

Checkpoint configuration:

- 8 encoder layers
- 8 decoder layers
- 6 attention heads
- Model dimension: 512
- Feed-forward dimension: 1,024
- Vocabulary size: 250,112 tokens
- Encoder-decoder sequence-to-sequence design

### Slide 3: Training data

The training notebook used Burmese news-style summarization data from the XL-Sum dataset:

- Training examples: 4,569
- Validation examples: 570
- Language: Burmese
- Training objective: Generate a summary from an article
- Training duration: 4 epochs
- Learning rate: `3e-4`
- Training batch size: 4
- Gradient accumulation: 8 steps
- Mixed precision: disabled during the final training configuration

### Slide 4: Burmese preprocessing

The model uses Burmese syllable segmentation before tokenization.

The preprocessing pipeline:

1. Trim leading and trailing whitespace.
2. Normalize repeated whitespace.
3. Separate Burmese syllable boundaries using the training regular expression.
4. Add the task prefix:

```text
summarize:
```

5. Tokenize the processed text.

Using the same preprocessing during inference helps keep the app input format consistent with the format used during training.

### Slide 5: Input and output limits

- Maximum article input: 512 tokens
- Default maximum summary length: 128 tokens
- Default minimum summary length: 20 tokens
- Long articles are truncated from the right when they exceed the input limit.
- Generated token spacing is removed before the Burmese summary is displayed.

### Slide 6: Text generation

The interface uses beam search to compare several possible summary sequences before selecting an output.

Default generation settings:

```text
Maximum output length: 128
Minimum output length: 20
Beam count: 4
Length penalty: 2.0
No-repeat n-gram size: 3
```

These controls allow users to trade off summary length, search quality, speed, and repetition control.

### Slide 7: Device and deployment

- CUDA is selected automatically when available.
- CPU is used automatically when CUDA is unavailable.
- The tokenizer and model are loaded once and reused for later requests.
- No model is downloaded from Hugging Face when local checkpoint files are present.
- The interface binds only to `127.0.0.1:7860` by default.

### Slide 8: Model strengths

- Works directly with Burmese news-style articles.
- Produces Burmese summaries without translation to another language.
- Uses the same segmentation and task prefix as the training pipeline.
- Runs privately on a local computer.
- Supports both GPU and CPU inference.

### Slide 9: Model limitations

- The output is an AI-generated draft, not a verified fact sheet.
- Important names, dates, numbers, and claims should be checked manually.
- Long articles may lose information after truncation.
- CPU generation can be slow for long articles or high beam counts.
- Summary quality depends on the fine-tuning data and checkpoint quality.
- The model can omit details, repeat ideas, or produce grammatically imperfect Burmese.

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
