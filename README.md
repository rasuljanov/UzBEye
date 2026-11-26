# UzBEye: Exploring Accuracy–Efficiency Trade-offs in Uzbek Visual Grounding

<img src="frontend/public/uzbeye-mark.svg" alt="UzBEye logo" width="80">

UzBEye takes an image and a short Uzbek description such as *chapdagi ko‘k soyabon* and draws a box around the object it refers to. The local Vue demo lets you upload an image, switch between four models, and inspect ranked box candidates.

## Platform

![UzBEye demo showing two separate prediction boxes for the Uzbek prompt ikkita odam](docs/platform-demo.png)

Model C grounds *ikkita odam* (“two people”) and displays a box for each person in the Portrait sample.

## Motivation

A pretrained detector can ground English descriptions through translation at inference time. That adds latency to each new query, and a translation error can change which object the user refers to. UzBEye studies how much accuracy is retained when the system learns to handle Uzbek prompts directly.

| Model | Approach |
|---|---|
| A · Translate-at-Inference | Translate Uzbek to English with NLLB, then run pretrained Grounding DINO-Tiny. |
| B · Translate-Train | Keep the original Grounding DINO text encoder and fine-tune on Uzbek RefCOCOg prompts. |
| C · Uzb-lang encoder | Replace the text encoder with TahrirchiBERT and fine-tune on the same Uzbek prompts as B. |
| D · Zero-Shot Uzbek | Send Uzbek prompts directly to pretrained Grounding DINO-Tiny. |

## Data and results

The project uses the [RefCOCOg UMD split](https://github.com/lichengunc/refer#how-to-use) and COCO images. The bilingual manifest contains **4,085 training** and **255 validation expressions**. The translation audit revised **3,328 of 4,340** prompts and preserved the image, target box, and split for every expression. It combined local machine checks with direct review of 1,444 prompts, including all validation prompts. The local audit (`data/refcocog/translation_audit/comparison.json`) records the English source, original translation, and revision by expression ID.

All rows below use the same **255 corrected validation prompts** on a Quadro RTX 5000 at batch size one. Top-1 accuracy requires the first box to overlap the target with IoU ≥ 0.5. Top-5 recall checks whether any of the first five boxes meets that threshold. Median latency includes translation for A and is measured after warm-up.

| Model and training data | Top-1 accuracy | Top-5 recall | Median latency |
|---|---:|---:|---:|
| A · Translate-at-Inference, uncached | 49.4% (126/255) | 82.0% | 450 ms |
| B · Original translations | 38.8% (99/255) | 85.1% | 208 ms |
| B · Corrected translations | 42.7% (109/255) | 81.6% | 207 ms |
| C · Original translations | 42.4% (108/255) | 76.9% | 207 ms |
| C · Corrected translations | 41.6% (106/255) | 82.7% | 204 ms |
| C · Corrected 4k + ranking loss | 48.2% (123/255) | 58.4% | 208 ms |
| C · Polished 8k + ranking loss | **56.5%** (144/255) | 64.7% | 236 ms |
| D · Zero-Shot Uzbek | 35.3% (90/255) | 65.9% | 212 ms |

See the [benchmark data and evaluation protocol](docs/validation-results.json). The validation split was used for checkpoint selection. The 128 selected official test expressions still have English annotations and need Uzbek review before held-out evaluation.

### Reading the results

![Scatter plot with a thin line connecting the four Model C runs. Polished 8k C reaches 56.5% at 236 ms.](docs/figures/accuracy-latency.png)

**Accuracy and latency.** Polished 8k C returned 144 correct first boxes at 236 ms median latency. Translate-at-Inference returned 126 at 450 ms, including uncached translation.

![Top-1 accuracy and top-5 recall with a thin line connecting all four Model C checkpoints.](docs/figures/training-runs.png)

**Translation repairs.** B gained 10 first-ranked matches and lost nine top-five matches. C lost two first-ranked matches and gained 15 top-five matches. The training prompts and the validation prompts used for checkpoint selection both changed between the two versions.

![Stacked bars of first-match ranks. Polished 8k C has 144 rank-one matches and 165 matches somewhere in the top five.](docs/figures/box-rank-breakdown.png)

**Ranking and data size.** The corrected 4k ranking-loss run reached 48.2% top-1 and 58.4% top-5. The polished 8k follow-up reached 56.5% top-1 and 64.7% top-5. It used more prompts, additional translation edits, and 5,000 optimizer updates compared with 2,555 for the 4k ranking run. Both checkpoints were selected using this validation split.

After installing the Python and frontend dependencies, regenerate the figures with `.\.venv\Scripts\python.exe -m scripts.plot_validation_results`.

## Why the results may differ

- **A can reuse pretrained English grounding.** When NLLB preserves the object, attributes, and spatial cues, Grounding DINO already has an alignment between those English features and image regions. Uncached translation accounts for much of A’s extra latency.
- **B starts with its pretrained text encoder and projection.** C starts with an Uzbek-language encoder and a fresh text projection. Its text features need to be aligned with the detector using the same 4,085 training expressions. This is a plausible difference in learning difficulty. B’s three-prediction lead over C on the corrected runs is small.

## Run locally

From the project folder, start the backend. The Uzbek encoder option loads `runs/polished-8k-ranking/C` when available, with `runs/corrected/C` as a fallback. Model B loads from `runs/corrected/B`.

```powershell
.\scripts\serve.ps1
```

The backend also serves the built frontend at [127.0.0.1:8000](http://127.0.0.1:8000). For live Vue development, leave the backend running and open a second terminal in the same project folder.

```powershell
npm --prefix frontend run dev
```

Open [127.0.0.1:5173](http://127.0.0.1:5173). Vite forwards API requests to port 8000. Set `GROUNDING_CHECKPOINTS` before starting the backend to use a different run folder.

For a fresh Windows setup, use Python 3.11, Node.js 24, and an NVIDIA driver compatible with CUDA 11.8.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
```

Try *ikki odam*, *barcha odamlar*, or *odamlar* for multiple matches. The demo filters overlapping duplicate boxes and displays up to ten matches together. A requested count caps the number of displayed boxes. Single-object expressions retain the ranked view.

Use the download icon in a model row to save its weights. A and D share the Grounding DINO-Tiny file. B and C download the adapter checkpoints selected by the backend. The adapter files require the pretrained detector. C also uses TahrirchiBERT, and A uses NLLB at inference time.

## Reproduce or extend

The commands below use the local corrected manifest. A fresh clone contains source and sample images. Prepare the dataset first, then provide a corrected manifest or use the generated `training_bilingual.json` for a run on the original translations. Each run saves its best validation checkpoint and training log. Set `GROUNDING_CHECKPOINTS` to serve checkpoints from a specific run folder.

```powershell
.\scripts\finetune.ps1 -Model B -RunName reproduction -Manifest data/refcocog/training_bilingual_corrected.json
.\scripts\finetune.ps1 -Model C -RunName reproduction -Manifest data/refcocog/training_bilingual_corrected.json
.\.venv\Scripts\python.exe -m scripts.run_comparison --root data/refcocog --manifest data/refcocog/training_bilingual_corrected.json --runs runs/reproduction --out runs/reproduction-validation --validation
```

To test Model C with the existing loss plus a box ranking loss, use the same corrected manifest and a fresh run name.

```powershell
.\scripts\finetune.ps1 -Model C -RunName corrected-ranking -Manifest data/refcocog/training_bilingual_corrected.json -RankingLoss
```

The launcher starts from the same pretrained weights as the reproduction run. Each validation pass prints top-1 and top-5 at IoU ≥ 0.5. The best checkpoint is selected by top-1, and its paired top-5 is recorded in `runs/corrected-ranking/C/training.json`.

### Expanded 8k training set

The expanded manifest keeps the 4,085 corrected training prompts and adds 3,915 RefCOCOg expressions. All 255 validation rows are identical to the 4k benchmark. The 8,000 training prompts span 7,090 distinct images, including 3,327 new images. In 588 scenes, the additions give the model distinct targets of the same category.

```powershell
.\scripts\finetune.ps1 -Model C -RunName corrected-8k -Manifest data/refcocog/training_bilingual_8k.json
.\scripts\finetune.ps1 -Model C -RunName corrected-8k-ranking -Manifest data/refcocog/training_bilingual_8k.json -RankingLoss
```

The expansion compares two Uzbek translations per English source. It uses two English round-trip checks, tests for identifying cues such as position, color, and count, and directly checks a subset of difficult examples against their English sources. The local `data/refcocog/expansion_8k/` directory holds the candidate translations, direct corrections, selection decisions, and dataset summary. Benchmarks based on this manifest report inference latency. Preprocessing cost and cost totals that depend on it require measured per-prompt translation timings.

To rebuild from the prepared 4k workspace, run `python -m scripts.expand_dataset` with `--stage candidates`, `--stage audit`, and `--stage finalize` in that order. Finalization downloads the selected COCO images and verifies split isolation, prompt length for both tokenizers, and image dimensions and decoding.

### Review Uzbek captions against the images

`scripts.polish_dataset` compares each English prompt and existing Uzbek caption with the annotated image. It saves decisions as it works, so the first command can be rerun after an interruption. Finalization checks rewritten captions by back-translation, preserves the source and box metadata, and writes the separate image-reviewed manifest. It saves bilingual decisions under `data/refcocog/polish_8k/`.

Install and start Ollama, then pull the local review model before running the first two commands.

```powershell
ollama pull qwen3-vl:4b-instruct
.\.venv\Scripts\python.exe -m scripts.polish_dataset --stage review
.\.venv\Scripts\python.exe -m scripts.polish_dataset --stage finalize
.\scripts\finetune.ps1 -Model C -RunName polished-8k-ranking -Manifest data/refcocog/training_bilingual_8k_polished.json -RankingLoss
```

The image-reviewed manifest keeps the 255 previously reviewed validation rows byte-for-byte so the new run can be compared with the earlier validation scores.

To prepare the original subset from scratch, run `scripts.prepare_dataset` with `--root data/refcocog --limits 4096 256 128`, then `src.data.translate` with `--root data/refcocog --stage training --batch-size 16` using `python -m`. Increase the limits to select more expressions. The corrected manifest and its audit are stored separately from this generated baseline.

The Vue app lives in `frontend/`. The API, data pipeline, models, and evaluation live in `src/`, with launchers in `scripts/` and settings in `configs/`. Data, model caches, and run outputs are local artifacts covered by `.gitignore`.

## Credits

[Grounding DINO-Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) · [TahrirchiBERT-base](https://huggingface.co/tahrirchi/tahrirchi-bert-base) · [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) · [Sample image credits](frontend/SAMPLES.md). The NLLB model card lists a noncommercial license.
