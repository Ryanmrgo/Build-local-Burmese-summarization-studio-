"""Run the local Burmese article summarization interface."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from summarizer import BurmeseSummarizer
from ui import create_demo


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Burmese article summarization interface."
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("BURMESE_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="Directory containing the local Hugging Face checkpoint.",
    )
    parser.add_argument("--port", type=int, default=7860, help="Local Gradio port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarizer = BurmeseSummarizer(Path(args.model_dir).expanduser().resolve())
    demo = create_demo(summarizer)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        inbrowser=False,
        share=False,
    )


if __name__ == "__main__":
    main()
