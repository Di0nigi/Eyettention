import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
import numpy as np




from Eyettention.raw_text_inference import EyettentionRawTextInference  

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BSC_CHECKPOINT = os.environ.get(
    "EYETTENTION_BSC_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "checkpoints", "bsc.pt"),
)
CELER_CHECKPOINT = os.environ.get(
    "EYETTENTION_CELER_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "checkpoints", "celer.pt"),
)

_MODELS: Dict[str, EyettentionRawTextInference] = {}


def get_model(dataset: str) -> EyettentionRawTextInference:
    """Load and cache an Eyettention model for the given dataset."""
    if dataset not in _MODELS:
        if dataset == "BSC":
            ckpt = BSC_CHECKPOINT
        elif dataset == "celer":
            ckpt = CELER_CHECKPOINT
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt!r}. "
                f"Set EYETTENTION_{dataset.upper()}_CHECKPOINT to the correct path."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODELS[dataset] = EyettentionRawTextInference(
            checkpoint_path=ckpt,
            dataset=dataset,
            device=device,
        )
    return _MODELS[dataset]



def words_for_text(text: str, dataset: str) -> List[str]:
    """
    Return the list of words that the model's fixation indices refer to.

    - For BSC (Chinese): the model tokenizes per character, so we return the
      list of characters in the input (excluding whitespace).
    - For CELER (English): the model tokenizes with a BERT wordpiece tokenizer
      on whitespace-split words, so we return the whitespace-split words.
    """
    if dataset == "BSC":
        
        return [ch for ch in text if not ch.isspace()]
    else:
        return text.split()


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------
def decode_scanpath(
    scanpath_indices: torch.Tensor,
    words: List[str],
    dataset: str,
) -> List[Dict[str, Any]]:
    """
    Convert a single predicted scanpath (indices into the word/char sequence)
    into a list of dicts describing each fixation.

    Convention in Eyettention's `scanpath_generation`:
      * index 0 corresponds to the CLS token (sentence start),
      * indices 1..len(words) correspond to words/characters,
      * index len(words)+1 corresponds to SEP (sentence end).

    We drop the leading CLS and any trailing SEP here.
    """
    if isinstance(scanpath_indices, torch.Tensor):
        scanpath_indices = scanpath_indices.detach().cpu().tolist()

    fixations: List[Dict[str, Any]] = []
    n_words = len(words)

    for step, idx in enumerate(scanpath_indices):
        idx = int(idx)

        if step == 0 and idx == 0:
            continue

        if idx == 0:
            word = "<CLS>"
        elif idx == n_words + 1:
            word = "<SEP>"
        elif 1 <= idx <= n_words:
            word = words[idx - 1]
        else:
            word = f"<OOR:{idx}>" 

        fixations.append(
            {
                "step": len(fixations) + 1,
                "word": word,
                "word_index": idx,
            }
        )

    return fixations


def fixations_to_markdown(
    fixations: List[Dict[str, Any]],
    original_text: str,
    dataset: str,
) -> str:
    """Render a scanpath as a Markdown table."""
    lines = [
        f"**Input ({dataset}):** {original_text}",
        "",
        "| Step | Fixated unit | Index |",
        "|------|--------------|-------|",
    ]
    for f in fixations:
        lines.append(f"| {f['step']} | {f['word']} | {f['word_index']} |")
    return "\n".join(lines)



def predict(
    text: str,
    dataset: str,
    max_pred_len: int,
    use_previous_scanpath: bool,
    previous_scanpath: str,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str]:
    """
    Run Eyettention on the input text and return:
      - a Markdown table of the predicted scanpath,
      - a space-separated string of the predicted word/char sequence,
      - the raw JSON of the scanpath indices and fixated units.
    """
    if text is None or text.strip() == "":
        raise gr.Error("Please provide some input text.")

    if dataset == "celer" and not any(c.isalpha() for c in text):
        raise gr.Error("CELER is the English model — please provide English text.")
    if dataset == "BSC" and not any("\u4e00" <= c <= "\u9fff" for c in text):
        
        pass

    if max_pred_len <= 0:
        raise gr.Error("max_pred_len must be a positive integer.")

    progress(0.05, desc="Loading Eyettention model...")
    model = get_model(dataset)

    prev: Optional[List[int]] = None
    if use_previous_scanpath and previous_scanpath.strip():
        try:
            prev = [int(x) for x in previous_scanpath.replace(",", " ").split()]
        except ValueError:
            raise gr.Error(
                "previous_scanpath must be a whitespace/comma-separated list of integers."
            )

    progress(0.25, desc="Running autoregressive scanpath generation...")
    with torch.no_grad():
        if dataset == "BSC":
            scanpath, _density = model.generate_from_chinese_text(
                text=text,
                max_pred_len=max_pred_len,
                previous_scanpath=prev,
            )
        else:
            scanpath, _density = model.generate_from_english_text(
                text=text,
                max_pred_len=max_pred_len,
                previous_scanpath=prev,
            )


    scanpath = scanpath[0]

    progress(0.9, desc="Formatting output...")
    words = words_for_text(text, dataset)
    fixations = decode_scanpath(scanpath, words, dataset)

    markdown = fixations_to_markdown(fixations, text, dataset)
    word_seq = " ".join(f["word"] for f in fixations)
    json_out = json.dumps(
        {
            "dataset": dataset,
            "input": text,
            "scanpath_indices": [int(i) for i in scanpath.detach().cpu().tolist()],
            "fixated_units": [f["word"] for f in fixations],
        },
        ensure_ascii=False,
        indent=2,
    )

    return markdown, word_seq, json_out


DESCRIPTION = """
# Eyettention — Scanpath Prediction

**Eyettention** predicts human-like **eye-movement scanpaths** from raw text.
Given a sentence, it autoregressively generates a sequence of fixation
locations (word or character indices) that approximate where a reader would
look, in order.

Two checkpoints are supported:

| Dataset | Language | Tokenization |
|---------|----------|--------------|
| **BSC** | Chinese | Character-level |
| **CELER** | English | Word-level (BERT wordpieces pooled) |

### How to use
1. Choose the **dataset / language**.
2. Paste your text.
3. Adjust `max_pred_len` if you want shorter or longer scanpaths.
4. Click **Run**.

### Optional: replay an observed prefix
Enable **Use previous scanpath** and provide a whitespace- or comma-separated
list of integer fixation indices. Those fixations will be replayed before the
model starts sampling new ones (useful for conditional generation / prefix
completion).
"""

EXAMPLES = [
    ["The quick brown fox jumps over the lazy dog.", "celer", 20, False, ""],
    ["今天天气很好，我们去公园散步吧。", "BSC", 20, False, ""],
]


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Eyettention — Scanpath Prediction",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                text_in = gr.Textbox(
                    label="Input text",
                    placeholder="Enter a sentence...",
                    lines=5,
                )
                dataset_in = gr.Radio(
                    choices=["celer", "BSC"],
                    value="celer",
                    label="Dataset / language",
                    info="celer = English, BSC = Chinese",
                )
                max_len_in = gr.Slider(
                    minimum=1,
                    maximum=120,
                    value=30,
                    step=1,
                    label="max_pred_len",
                    info="Maximum number of fixations to generate.",
                )
                with gr.Accordion("Advanced: replay an observed prefix", open=False):
                    use_prev_in = gr.Checkbox(
                        value=False,
                        label="Use previous scanpath",
                    )
                    prev_in = gr.Textbox(
                        label="Previous scanpath (integer indices)",
                        placeholder="e.g. 0 1 3 5 4",
                        lines=2,
                    )
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=4):
                table_out = gr.Markdown(
                    label="Predicted scanpath",
                    value="_Results will appear here._",
                )
                seq_out = gr.Textbox(
                    label="Fixated units (sequence)",
                    lines=3,
                    interactive=False,
                )
                json_out = gr.Code(
                    label="Raw JSON output",
                    language="json",
                    value="{}",
                )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_in, dataset_in, max_len_in, use_prev_in, prev_in],
        )

        def _run(text, dataset, max_pred_len, use_prev, prev):
            md, seq, js = predict(
                text=text,
                dataset=dataset,
                max_pred_len=int(max_pred_len),
                use_previous_scanpath=use_prev,
                previous_scanpath=prev,
            )
            return md, seq, js

        run_btn.click(
            fn=_run,
            inputs=[text_in, dataset_in, max_len_in, use_prev_in, prev_in],
            outputs=[table_out, seq_out, json_out],
        )
        clear_btn.click(
            fn=lambda: ("", "celer", 30, False, "", "_Results will appear here._", "", "{}"),
            inputs=None,
            outputs=[
                text_in,
                dataset_in,
                max_len_in,
                use_prev_in,
                prev_in,
                table_out,
                seq_out,
                json_out,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(max_size=16).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )