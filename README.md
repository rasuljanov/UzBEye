# UzBEye: Exploring Accuracy–Efficiency Trade-offs in Uzbek Visual Grounding

<img src="frontend/public/uzbeye-mark.svg" alt="UzBEye logo" width="80">

UzBEye takes an image and a short Uzbek description such as *chapdagi ko‘k soyabon* and draws a box around the object it refers to. The local Vue demo lets you upload an image, switch between four models, and inspect their five highest-ranked boxes.

## Motivation

A pretrained detector can ground English descriptions through translation at inference time. That adds latency to each new query, and a translation error can change which object the user refers to. UzBEye studies how much accuracy is retained when the system learns to handle Uzbek prompts directly.

| Model | Approach |
|---|---|
| A · Translate-Test | Translate Uzbek to English with NLLB, then run pretrained Grounding DINO-Tiny. |
| B · Translate-Train | Keep the original Grounding DINO text encoder and fine-tune on Uzbek RefCOCOg prompts. |
| C · TahrirchiBERT | Replace the text encoder with TahrirchiBERT and fine-tune on the same Uzbek prompts as B. |
| D · Zero-Shot Uzbek | Send Uzbek prompts directly to pretrained Grounding DINO-Tiny. |

## Data and results

The project uses the [RefCOCOg UMD split](https://github.com/lichengunc/refer#how-to-use) and COCO images. The bilingual manifest contains **4,085 training** and **255 validation expressions**. The translation audit revised **3,328 of 4,340** prompts and preserved the image, target box, and split for every expression. It combined local machine checks with direct review of 1,444 prompts, including all validation prompts. The local audit (`data/refcocog/translation_audit/comparison.json`) records the English source, original translation, and revision by expression ID.

All rows below use the same **255 corrected validation prompts** on a Quadro RTX 5000 at batch size one. Top-1 accuracy requires the first box to overlap the target with IoU ≥ 0.5. Top-5 recall checks whether any of the first five boxes meets that threshold. Median latency includes translation for A and is measured after warm-up.

| Model and training data | Top-1 accuracy | Top-5 recall | Median latency |
|---|---:|---:|---:|
| A · Translate-Test, uncached | **49.4%** (126/255) | 82.0% | 450 ms |
| B · Original translations | 38.8% (99/255) | 85.1% | 208 ms |
| B · Corrected translations | 42.7% (109/255) | 81.6% | 207 ms |
| C · Original translations | 42.4% (108/255) | 76.9% | 207 ms |
| C · Corrected translations | 41.6% (106/255) | 82.7% | 204 ms |
| D · Zero-Shot Uzbek | 35.3% (90/255) | 65.9% | 212 ms |

See the [benchmark data and evaluation protocol](docs/validation-results.json). The validation split was used for checkpoint selection. The 128 selected official test expressions still have English annotations and need Uzbek review before held-out evaluation.

## Why the results may differ

- **A can reuse pretrained English grounding.** When NLLB preserves the object, attributes, and spatial cues, Grounding DINO already has an alignment between those English features and image regions. Uncached translation accounts for much of A’s extra latency.
- **B starts with its pretrained text encoder and projection.** C starts with an Uzbek-language encoder and a fresh text projection. Its text features need to be aligned with the detector using the same 4,085 training expressions. This is a plausible difference in learning difficulty. B’s three-prediction lead over C on the corrected runs is small.
- **Translation corrections affected ranking and candidate coverage differently.** B gained 10 first-ranked matches after retraining. C found a matching box in its top five on 15 more prompts, while it lost two first-ranked matches. The training captions and checkpoint-selection prompts were both corrected, so the comparison describes the two complete training runs.

## Run locally

From the project folder, start the backend. It loads B and C from `runs/corrected` by default.

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

## Reproduce or extend

The commands below use the local corrected manifest. A fresh clone contains source and sample images. Prepare the dataset first, then provide a corrected manifest or use the generated `training_bilingual.json` for a run on the original translations. Each run saves its best validation checkpoint and training log. The tester loads B and C once their checkpoints are available in `runs/corrected` or in the folder set by `GROUNDING_CHECKPOINTS`.

```powershell
.\scripts\finetune.ps1 -Model B -RunName reproduction -Manifest data/refcocog/training_bilingual_corrected.json
.\scripts\finetune.ps1 -Model C -RunName reproduction -Manifest data/refcocog/training_bilingual_corrected.json
.\.venv\Scripts\python.exe -m scripts.run_comparison --root data/refcocog --manifest data/refcocog/training_bilingual_corrected.json --runs runs/reproduction --out runs/reproduction-validation --validation
```

To prepare the original subset from scratch, run `scripts.prepare_dataset` with `--root data/refcocog --limits 4096 256 128`, then `src.data.translate` with `--root data/refcocog --stage training --batch-size 16` using `python -m`. Increase the limits to select more expressions. The corrected manifest and its audit are stored separately from this generated baseline.

The Vue app lives in `frontend/`. The API, data pipeline, models, and evaluation live in `src/`, with launchers in `scripts/` and settings in `configs/`. Data, model caches, and run outputs are local artifacts covered by `.gitignore`.

## Credits

[Grounding DINO-Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) · [TahrirchiBERT-base](https://huggingface.co/tahrirchi/tahrirchi-bert-base) · [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) · [COCO sample credits](frontend/SAMPLES.md). The NLLB model card lists a noncommercial license.
