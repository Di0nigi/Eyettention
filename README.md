# Eyettention: An Attention-based Dual-Sequence Model for Predicting Human Scanpaths during Reading

In this paper, we develop Eyettention, the first dual-sequence model that simultaneously processes the sequence of words and the chronological sequence of fixations. The alignment of the two sequences is achieved by a cross-sequence attention mechanism. We show that Eyettention outperforms state-of-the-art models in predicting scanpaths. We provide an extensive within- and across-data set evaluation on different languages. An ablation study and qualitative analysis support an in-depth understanding of the model's behavior.

The overview above describes the [original authors' work](https://arxiv.org/abs/2304.10784). This project uses the [original Eyettention implementation](https://github.com/aeye-lab/Eyettention), reorganized as the `Eyettention` package, with raw-text inference, scanpath-prefix replay, an endpoint handler, and a Gradio interface.

## Setup

For this checkout, run setup and inference commands from the project root (the directory containing `Eyettention/`). Install dependencies:

```bash
python -m pip install -r Eyettention/requirements.txt
```

The requirements retain historical PyTorch and Transformers pins that may need adjustment for your Python version and platform. The local preprocessing code also requires `LAC` and a compatible PaddlePaddle installation; BSC Excel loading requires `openpyxl`. Install `gradio` if using the web interface. Use a separate environment from ScanDL2 because their dependency versions differ.

## Dataset

For the CELER dataset, follow the instructions at [the CELER repository](https://github.com/berzak/celer). In order to run experiments, place the downloaded dataset in `Eyettention/Data/celer/`. The local loaders expect `data_v2.0/sent_fix.tsv`, `data_v2.0/sent_ia.tsv`, and `metadata.tsv` beneath that directory.

The Chinese [Beijing Sentence Corpus (BSC)](https://osf.io/vr3k8/) belongs in `Eyettention/Data/beijing-sentence-corpus/`, with `BSC.Word.Info.v2.xlsx` and `BSC.EMD/BSC.EMD.txt`. See [utils.py](utils.py) for the dataset loaders.

Raw-text inference uses the trained checkpoints and normalization files already stored under `results/` and `Data/`; it does not require the full training corpora.

## Run Experiments

The experiment scripts use paths relative to `Eyettention/`. Starting at the project root:

```bash
cd Eyettention
mkdir -p results/BSC results/CELER
```

Keep the project root on `PYTHONPATH` when running the package modules.

### For Chinese BSC dataset

```bash
PYTHONPATH=.. python -m Eyettention.main_BSC --test_mode text --gpu 0
PYTHONPATH=.. python -m Eyettention.main_BSC --test_mode subject --gpu 0
PYTHONPATH=.. python -m Eyettention.main_BSC_NRS_setting --gpu 0
PYTHONPATH=.. python -m Eyettention.main_BSC_reader_identifier --gpu 0
```

### For English CELER dataset

```bash
PYTHONPATH=.. python -m Eyettention.main_celer --test_mode text --gpu 0
PYTHONPATH=.. python -m Eyettention.main_celer --test_mode subject --gpu 0
PYTHONPATH=.. python -m Eyettention.main_celer_NRS_setting --gpu 0
PYTHONPATH=.. python -m Eyettention.main_celer_reader_identifier --gpu 0
```

`text` evaluates new sentences; `subject` evaluates new readers. Select an available GPU with `--gpu`. Several historical experiment scripts call CUDA directly, so use a CUDA-enabled environment for these commands. Review output paths before running; `--save_data_folder` selects the results directory.

## Raw-text inference

Run the following from the project root:

```python
from Eyettention import EyettentionRawTextInference

model = EyettentionRawTextInference(
    checkpoint_path="Eyettention/results/CELER/Eyettention_english.pth",
    dataset="celer",
    device="cpu",
)
scanpath, density = model.generate_from_english_text(
    "The quick brown fox jumps.",
    max_pred_len=20,
    # Optional: replay observed fixation positions before sampling.
    # previous_scanpath=[0, 1, 2],
)
print(scanpath[0].tolist())
```

For Chinese, use `dataset="BSC"`, checkpoint `Eyettention/results/BSC/Eyettention_chinese.pth`, and `generate_from_chinese_text(...)`. Dataset names are case-sensitive. Use `device="cuda"` for GPU inference. BERT assets must be downloadable from Hugging Face or cached locally.

Keep `Data/feature_norm_celer.pickle` and `Data/feature_norm_BSC.pickle` alongside the package. The output contains fixation indices and per-step probability distributions, not fixation durations. Index `0` denotes CLS, positions `1..N` refer to words or Chinese characters, and `N+1` denotes SEP; stop at the first SEP when interpreting fixations. Use short sentences within the checkpoint's configured input limits. Prefix replay extends the generation loop, so `max_pred_len` is not a strict total-length cap when a prefix is provided.

## Endpoint handler

The added [handler.py](handler.py) wraps inference in a dictionary-based interface:

```python
from Eyettention.handler import EndpointHandler

handler = EndpointHandler()
result = handler({
    "inputs": "The quick brown fox jumps.",
    "parameters": {
        "dataset": "celer",
        "device": "cpu",
        "max_pred_len": 20,
    },
})
```

It returns `scanpath` and `density_steps`. Optional parameters include `checkpoint_path` and `previous_scanpath`. The handler defaults to BSC on CPU and does not itself start an HTTP server.

## Gradio interface

The added [app.py](app.py) provides a web interface. From the project root, point it to the local checkpoints and launch:

```bash
export EYETTENTION_BSC_CHECKPOINT="Eyettention/results/BSC/Eyettention_chinese.pth"
export EYETTENTION_CELER_CHECKPOINT="Eyettention/results/CELER/Eyettention_english.pth"
python -m Eyettention.app
```

Open `http://localhost:7860`, choose the dataset, enter text, and click **Run**. The interface also supports an observed scanpath prefix. It selects CUDA when available, otherwise CPU. Set `PORT` to change the default port; the app binds to `0.0.0.0`.

## Cite our work

If you use this code for your research, please consider citing the original authors' paper:

```bibtex
@article{deng2023eyettention,
  title={Eyettention: {A}n Attention-based Dual-Sequence Model for Predicting Human Scanpaths during Reading},
  author={Deng, Shuwen and Reich, David R and Prasse, Paul and Haller, Patrick and Scheffer, Tobias and J{\"a}ger, Lena A},
  journal={Proceedings of the {ACM} on Human-Computer Interaction},
  volume={7},
  number={ETRA},
  pages={1--24},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

## License

The original code is provided under the [MIT License](LICENSE.txt), copyright © 2023 AEye. Include the copyright and permission notice when redistributing copies or substantial portions of the software. See the license file for the full terms and warranty disclaimer. Datasets and dependencies retain their own licenses.
