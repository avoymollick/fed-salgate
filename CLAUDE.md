# Fed-SalGate — Project Reference

**Thesis title:** FedSalGate: Federated Saliency-Consensus Gating for Drone-Based Flood Damage Assessment
**Institution:** AIUB · **Domain:** Deep Learning / Federated Learning / Explainable AI
**Group:** Md. Nawshin Navin (23-50079-1), Avoy Mollick (23-50066-1), Nafiun Apurbo (23-50064-1), Arpita Barua (22-49827-3)
**Repo:** git@github.com:avoymollick/fed-salgate.git (private)

See `PROGRESS.md` for the dated run log / results history. This file is the stable reference — update it only when something structural changes (new part, new convention, new gotcha), not after every run.

## What this is

A federated learning system for flood-damage semantic segmentation from drone/aerial imagery (FloodNet dataset). Three simulated regional clients train a shared segmentation model without sharing raw photos. The core contribution, **Fed-SalGate**, replaces uniform FedAvg-style aggregation with **saliency-consensus gating**: the server computes each client's Grad-CAM explanation on a shared probe set, compares it against the mean of the others (soft-IoU), and down-weights clients whose explanations disagree — `sigmoid(β·(agreement−τ))`. Benchmarked against FedAvg, FedProx, and FedSeg (Miao et al., CVPR 2023), plus centralized literature baselines (UNet/SegNet/DeepLabV3+ variants, FloodNet-Lite).

The official university submission is a short high-level paragraph (FL + saliency-guided attention + XAI, non-IID robustness) — it does **not** commit us to the much larger scope in `docs/Fed_SalGate_Thesis_Workflow_Report.pdf` (an AI-generated 48-week plan with xBD/BRIGHT datasets, SAR fusion, secure aggregation, 5-20 clients, etc.). That report is scope inspiration only — treat the actual notebook pipeline below as the real target.

## Repo structure

```
part1/  fed-salgate-v2-part-1.ipynb      — pseudo-labeling + self-training + non-IID partition (EXECUTED on Kaggle)
part2/  notebook3896666d14.ipynb          — Fed-SalGate vs FedAvg vs FedProx + LiteUNet (the core contribution)
part3/  part3_01..06_*.ipynb              — 6 centralized literature baselines (UNet/SegNet/DeepLabV3+ x ResNet34/EfficientNet-B0, FloodNet-Lite)
        part3_07_fedseg_deeplabv3p.ipynb  — FedSeg federated baseline (Miao et al. CVPR 2023)
        part3_99_merge_results.ipynb      — collects all result_*.json + part2 summary into comparison tables/plots/Excel
docs/   Fed_SalGate_Thesis_Workflow_Report.pdf, SUMMER_Thesis_Group.docx
```

Each Part 3 model notebook is designed to run standalone and emit `result_<model>.json`; the merge notebook (`part3_99`) collects them all. This was originally built so group members could split compute across separate Kaggle accounts — no longer needed once running on RunPod.

## Dataset

- **FloodNet Track 1** (Kaggle: `aletbm/aerial-imagery-dataset-floodnet-challenge`) — 10-class semantic segmentation. Classes: Background, Building-Flooded, Building-NonFlooded, Road-Flooded, Road-NonFlooded, Water, Tree, Vehicle, Pool, Grass. 398 labeled (51 flooded + 347 non-flooded), 1047 unlabeled train, 450 val, 448 test.
- **290-image binary water set** (Kaggle: `faizalkarim/flood-area-segmentation`) — merged in as `water=class 5, everything else=ignore(255)` to sharpen the Water class.
- **Held-out GT protocol**: 15% of real labeled photos (59 of 398) are carved out once in Part 1 and never touched by self-training or FL client partitions. All final evaluation numbers use this held-out set — this is the "honest evaluation" guarantee referenced throughout.

## Key config (from Part 1's `config.json`, as actually run on Kaggle)

`IMG_SIZE=512, BATCH_SIZE=4, NUM_CLIENTS=3, NUM_ROUNDS=50, LOCAL_EPOCHS=3, DIRICHLET_ALPHA=1.5, LR=1e-4, PROXIMAL_MU=0.01, SALIENCY_LAMBDA=0.1, SELFTRAIN_EPOCHS=40`. Client split: 852/333/201 samples (mild non-IID — all 3 Grass-dominant).

**Pending decision:** bump self-training to 100 epochs + early stopping (patience~15), and consider a more extreme non-IID re-partition (5 clients, α=0.3) before the real RunPod run — see PROGRESS.md.

## Environment

- **Local laptop**: Windows, RTX 4050 Laptop GPU (6GB), Python 3.11 venv at `a:/thesis/.venv` (torch 2.5.1+cu121). Used for fast demo/smoke-test runs only, not final results.
- **Local data staging** (gitignored, not in repo): `a:/thesis/data/floodnet/`, `a:/thesis/data/water290/`, `a:/thesis/data/part1_outputs/` (masks/checkpoint/config pulled from Navin's Kaggle kernel via `kaggle kernels output navinxqz/fed-salgate-v2-part-1`).
- **RunPod**: planned for full-scale runs, on hold pending pod provisioning. SSH keypair already generated locally (`~/.ssh/runpod_fedsalgate`), public key ready to add to RunPod account settings.
- **Credentials**: Kaggle token at `~/.kaggle/kaggle.json`, GitHub via dedicated SSH key (`~/.ssh/github_fedsalgate`, configured in `~/.ssh/config`). HuggingFace token not yet set up.

## Known gotchas (read before editing notebooks)

1. **Part 2's notebook stores cell `source` as a single string**, not a list of lines (Part 3's notebooks use the standard list format). Any scripted find/replace across cells MUST handle both — join to a string, edit, then reconstruct the original type. Getting this wrong silently corrupts the file into a per-character array (still valid JSON, `''.join()` reads back fine, but the raw file bloats ~8x and byte-level substrings break). Always verify with `ast.parse()` per cell after patching, and check file size stays sane before committing.
2. **`num_workers` must be 0 on Windows** when running the notebook-converted `.py` script locally — the script isn't guarded with `if __name__ == "__main__":`, so `DataLoader(num_workers>0)` triggers Windows to respawn and re-execute the whole script, crashing with a multiprocessing bootstrap `RuntimeError`. The committed notebooks use `num_workers=4` (correct for RunPod/Linux); only the local-laptop-only patched copy (`part2/part2_run.py`, gitignored) overrides it back to 0.
3. **The "package outputs" zip step self-includes**: `shutil.make_archive(..., WORKING_DIR)` zips the entire working directory, including any zip from a previous run sitting inside it — reruns compound and can fill a disk (this happened once, 166GB, fixed by deleting stale zips before archiving). Fixed in all 9 notebooks as of 2026-08-08; the fix is a `glob.glob(*.zip) + os.remove()` cleanup right before `make_archive`. Preserve this if regenerating any of these cells.
4. **Python 3.14** (the system default `python`) is too new for current PyTorch wheels — always use the `a:/thesis/.venv` (Python 3.11) for anything touching torch.
5. Console `print()`/`grep` on these notebooks can hit `UnicodeEncodeError`/false-negative issues (emoji in markdown cells, compact-JSON files read as few giant "lines") — prefer writing diagnostic output to a file and reading it back, and use occurrence-counting (`text.count(...)`) rather than `grep -c` (which counts matching *lines*, misleading on compact JSON).
