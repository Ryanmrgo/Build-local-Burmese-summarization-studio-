"""Gradio interface for the local Burmese summarization model."""

from __future__ import annotations

import gradio as gr

from summarizer import BurmeseSummarizer


EXAMPLE_TEXT = (
    "ကုလသမဂ္ဂအထွေထွေအတွင်းရေးမှူးချုပ်က ရာသီဥတုပြောင်းလဲမှုနှင့်ပတ်သက်၍ "
    "နိုင်ငံတကာပူးပေါင်းဆောင်ရွက်ရန် တိုက်တွန်းလိုက်သည်။"
)

CUSTOM_CSS = """
:root {
    --ink: #172321;
    --muted: #6c7b76;
    --paper: #f7f4ed;
    --panel: #fffdf8;
    --line: #d9e1d8;
    --green: #1e6254;
    --green-dark: #12463d;
    --coral: #e2775f;
}
body, .gradio-container {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: "Noto Sans Myanmar", "Myanmar Text", sans-serif !important;
}
.gradio-container { max-width: 1180px !important; padding: 34px 28px 54px !important; }
.hero { background: var(--green-dark); border-radius: 22px; margin-bottom: 18px; overflow: hidden; padding: 36px 42px 32px; position: relative; }
.hero:after { border: 1px solid rgba(255, 255, 255, 0.18); border-radius: 50%; content: ""; height: 240px; position: absolute; right: -42px; top: -100px; width: 240px; }
.hero h1, .hero p { position: relative; z-index: 1; }
.hero h1 { color: #fffdf8 !important; font-family: "Noto Sans Myanmar", "Myanmar Text", sans-serif !important; font-size: clamp(2rem, 4vw, 2.8rem) !important; line-height: 1.3 !important; margin: 0 0 8px !important; }
.hero p { color: #cfe1d8 !important; line-height: 1.7; margin: 0 !important; max-width: 640px; }
.eyebrow { color: #f3a58f !important; font-size: 12px !important; font-weight: 700 !important; letter-spacing: 1.6px !important; margin-bottom: 12px !important; text-transform: uppercase; }
.meta-row { gap: 12px !important; margin-bottom: 18px !important; }
.meta-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; min-height: 70px; padding: 14px 16px !important; }
.meta-card p { color: var(--muted) !important; font-size: 13px !important; margin: 0 !important; }
.meta-card strong { color: var(--ink); font-size: 16px; }
.workspace { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 12px 30px rgba(30, 71, 60, 0.06); gap: 20px !important; padding: 20px !important; }
.workspace h3 { color: var(--ink) !important; margin: 0 0 4px !important; }
.workspace .hint { color: var(--muted) !important; font-size: 13px !important; line-height: 1.5; }
textarea, input { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
textarea { background: #fbfaf5 !important; border-color: var(--line) !important; line-height: 1.8 !important; min-height: 300px !important; }
textarea::placeholder, input::placeholder { color: #82908a !important; -webkit-text-fill-color: #82908a !important; opacity: 1 !important; }
textarea:disabled, input:disabled { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; opacity: 1 !important; }
textarea:focus { border-color: var(--green) !important; box-shadow: 0 0 0 2px rgba(30, 98, 84, 0.12) !important; }
button.primary, .primary-button { background: var(--green) !important; border: 0 !important; min-height: 46px; }
button.primary:hover, .primary-button:hover { background: var(--green-dark) !important; }
.action-row { gap: 10px !important; margin-top: 12px; }
.action-row button { min-height: 46px; }
.status-bar { background: #e0f2ef; border: 1px solid #b7ddd7; border-radius: 7px; color: var(--green-dark) !important; margin: 14px 0 0 !important; padding: 9px 12px; }
.status { color: var(--muted); min-height: 24px; }
.settings { border: 1px solid var(--line) !important; border-radius: 14px !important; margin-top: 18px; }
.control-row { gap: 16px !important; }
#footer { color: var(--muted); font-size: 12px !important; margin-top: 18px !important; text-align: center; }
@media (max-width: 760px) {
    .gradio-container { padding: 18px 12px 36px !important; }
    .hero { border-radius: 16px; padding: 28px 22px 26px; }
    .meta-row { gap: 8px !important; }
    .workspace { border-radius: 14px; gap: 14px !important; padding: 14px !important; }
    textarea { min-height: 230px !important; }
    .control-row { gap: 0 !important; }
}
"""


def create_demo(summarizer: BurmeseSummarizer) -> gr.Blocks:
    def run_summary(
        article: str,
        max_length: int,
        min_length: int,
        num_beams: int,
        length_penalty: float,
        no_repeat_ngram_size: int,
    ) -> tuple[str, str, str]:
        try:
            summary, status = summarizer.summarize(
                article,
                max_length,
                min_length,
                num_beams,
                length_penalty,
                no_repeat_ngram_size,
            )
            return summary, status, f"{len(summary):,} characters"
        except Exception as error:
            return "", f"Model error: {error}", "0 characters"

    def clear_fields() -> tuple[str, str, str, str, str]:
        return "", "", "0 characters", "", "0 characters"

    with gr.Blocks(title="Burmese Article Summarizer", css=CUSTOM_CSS, theme=gr.themes.Base()) as demo:
        gr.HTML(
            """
            <div class="hero">
                <div class="eyebrow">AI READING DESK</div>
                <h1>မြန်မာစာ အနှစ်ချုပ်</h1>
                <p>Turn a long Burmese article into a clear first draft for discussion, review, or classroom presentation.</p>
            </div>
            """
        )

        with gr.Row(elem_classes="meta-row"):
            gr.HTML('<div class="meta-card"><p>MODEL</p><strong>mT5-small</strong></div>')
            gr.HTML('<div class="meta-card"><p>TASK</p><strong>Burmese summarization</strong></div>')
            gr.HTML(f'<div class="meta-card"><p>RUNTIME</p><strong>Local {summarizer.device.type.upper()}</strong></div>')

        with gr.Row(equal_height=False, elem_classes="workspace"):
            with gr.Column(scale=1):
                gr.Markdown("### 01  Add an article")
                gr.Markdown("Paste Burmese text below, or load the example.", elem_classes="hint")
                article = gr.Textbox(
                    lines=15,
                    placeholder="Paste the Burmese article you want to summarize...",
                    show_label=False,
                )
                input_count = gr.Markdown("0 characters", elem_classes="status")
                with gr.Row(elem_classes="action-row"):
                    summarize_button = gr.Button("Summarize article", variant="primary", elem_classes="primary-button")
                    example_button = gr.Button("Load example")
                    clear_button = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=1):
                gr.Markdown("### 02  Review the summary")
                gr.Markdown("The locally loaded model will generate a concise Burmese draft.", elem_classes="hint")
                summary = gr.Textbox(lines=15, show_label=False, interactive=False)
                output_count = gr.Markdown("0 characters", elem_classes="status")

        with gr.Accordion("အဆင့်မြင့် မော်ဒယ်ဆက်တင်များ / Advanced settings", open=False, elem_classes="settings"):
            with gr.Row(elem_classes="control-row"):
                max_length = gr.Slider(32, 250, value=128, step=1, label="Maximum output length")
                min_length = gr.Slider(0, 100, value=20, step=1, label="Minimum output length")
                num_beams = gr.Slider(1, 8, value=4, step=1, label="Beam count")
            with gr.Row(elem_classes="control-row"):
                length_penalty = gr.Slider(0.1, 3.0, value=2.0, step=0.1, label="Length penalty")
                no_repeat_ngram_size = gr.Slider(0, 5, value=3, step=1, label="No-repeat n-gram size")

        status = gr.Markdown(
            "Runs privately on this computer. Generated text should be reviewed before teaching or publication.",
            elem_classes=["status", "status-bar"],
        )
        gr.Markdown("Local only · Your article is processed on this computer.", elem_id="footer")

        article.change(lambda value: f"{len(value or ''):,} characters", inputs=article, outputs=input_count)
        summarize_button.click(
            run_summary,
            inputs=[article, max_length, min_length, num_beams, length_penalty, no_repeat_ngram_size],
            outputs=[summary, status, output_count],
        )
        article.submit(
            run_summary,
            inputs=[article, max_length, min_length, num_beams, length_penalty, no_repeat_ngram_size],
            outputs=[summary, status, output_count],
        )
        example_button.click(lambda: EXAMPLE_TEXT, outputs=article)
        clear_button.click(
            clear_fields,
            outputs=[article, summary, input_count, status, output_count],
        )

    return demo
