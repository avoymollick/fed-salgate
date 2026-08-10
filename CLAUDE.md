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

## Key config

`IMG_SIZE=512, BATCH_SIZE=4, NUM_CLIENTS=3, NUM_ROUNDS=50, LOCAL_EPOCHS=3, DIRICHLET_ALPHA=1.5, LR=1e-4, PROXIMAL_MU=0.01, SALIENCY_LAMBDA=0.1`.

**Self-training (2026-08-09, RunPod rerun — this checkpoint/masks are the live foundation, reused as-is)**: `SELFTRAIN_EPOCHS=100` (was 40) + early stopping (patience=15 on held-out val mIoU), never triggered — ran the full 100 epochs. **Held-out val mIoU: 0.4117** (vs original 0.382). Archived to HF: `avoymollick/fedsalgate-artifacts/part1_100epoch`.

**Client partitioning — CHANGED as of 2026-08-10**: re-partitioned from the original `NUM_CLIENTS=3, DIRICHLET_ALPHA=1.5` to **`NUM_CLIENTS=5, DIRICHLET_ALPHA=0.3`** (client sizes 107/815/33/300/131) to test whether harsher non-IID would let saliency-consensus gating activate. Old 3-client/α=1.5 partition backed up (`client_indices.pkl.3clients_a1.5.bak` / `config.json.3clients_a1.5.bak`) alongside the current one in `/workspace/working/fed_salgate_outputs/` on the pod.

**Part 2 results, both non-IID regimes now tested — Fed-SalGate has not beaten FedAvg in either**:

| Split | FedAvg | FedProx | Fed-SalGate | Gates actually moved? |
|---|---|---|---|---|
| 3 clients, α=1.5 (mild) | 0.4229 | 0.4174 | 0.4186 | No — stayed ~uniform (0.98–1.02) |
| 5 clients, α=0.3 (harsh) | 0.4145 | 0.4071 | 0.4086 | **Yes** — cleanly correlated with agreement, lowest-agreement client got lowest gate |

So the gating mechanism demonstrably works as designed under real non-IID stress, but that correct behavior doesn't translate into better held-out generalization — if anything Fed-SalGate's gap vs FedAvg is slightly wider under the harsher split. **Nuance worth keeping**: Fed-SalGate wins specifically on Water IoU (0.5237 vs 0.4674/0.4758) in the 5-client run — meaningful for a flood-damage thesis even though overall mIoU is behind. See PROGRESS.md 2026-08-10 for full per-class breakdown and the open question of what this means for the paper's narrative — **not yet resolved, needs a decision before Part 3**.

**In progress (2026-08-10): gate-hyperparameter tuning sweep.** Diagnosed why gates barely moved even in the "activated" 5-client run: real agreement scores (0.769–0.930) are far above `τ=0.5`, so `sigmoid(β·(agreement−τ))` was always saturated near 1.0 regardless of relative differences between clients — a calibration bug, not a mechanism failure. Sweeping `(τ=0.85,β=10)`, `(τ=0.85,β=20)`, `(τ=0.80,β=15)`, training Fed-SalGate only per config (FedAvg=0.4145/FedProx=0.4071 reused from the cached 5-client/α=0.3 run, not retrained). Script: `part2/part2_gate_sweep.py` (pod-only so far, not yet backported to git). Results land in `/workspace/working/fed_salgate_outputs/gate_sweep_results.json`, written incrementally per config. See PROGRESS.md 2026-08-10 for the pod-resume snags hit along the way.

## Environment

- **Local laptop**: Windows, RTX 4050 Laptop GPU (6GB), Python 3.11 venv at `a:/thesis/.venv` (torch 2.5.1+cu121). Used for fast demo/smoke-test runs only, not final results.
- **Local data staging** (gitignored, not in repo): `a:/thesis/data/floodnet/`, `a:/thesis/data/water290/`, `a:/thesis/data/part1_outputs/` (Navin's original 40-epoch Kaggle outputs — superseded once the RunPod 100-epoch rerun finishes, kept for comparison).
- **RunPod** (STOPPED as of 2026-08-10, not terminated — volume/data intact, resume by starting the pod again, not redeploying): pod deployed via the `runpod` Python SDK (not the web UI) using an account-level API key — full deploy/manage access, not just SSH. Pod ID `0lrs9f0awbogvn`: RTX 4090, 24GB VRAM, Secure Cloud ($0.74/hr while running, no charge while stopped — Community had no stock at deploy time), `/workspace` volume. SSH alias `runpod` configured in `~/.ssh/config` (host/port will likely change on restart — call `runpod.get_pod('0lrs9f0awbogvn')` after starting it again to get the fresh `runtime.ports` mapping and update the alias). Repo cloned to `/workspace/fed-salgate`; datasets at `/workspace/data/floodnet`, `/workspace/data/water290`. **To resume**: `runpod.resume_pod('0lrs9f0awbogvn')` (or equivalent start call), wait for `desiredStatus: RUNNING`, re-fetch the SSH port, everything else (code, data, results) is already there.
- **Credentials** (all local-only, gitignored, never committed):
  - Kaggle: `~/.kaggle/kaggle.json` (laptop) and `~/.kaggle/kaggle.json` (pod, copied over)
  - GitHub: dedicated SSH key `~/.ssh/github_fedsalgate` (laptop), copied to the pod at the same path for the pod's own `git` access
  - RunPod: SSH key `~/.ssh/runpod_fedsalgate` (laptop only, private key never leaves the laptop); API key at `~/.runpod/api_key.txt` (laptop only) — used via the `runpod` pip package for pod create/terminate
  - HuggingFace: token at `~/.cache/huggingface/token` (laptop). Private dataset repo: `avoymollick/fedsalgate-artifacts` — the archive destination for generated results/checkpoints, not a dependency for starting a run (see PROGRESS.md 2026-08-09 for the full explanation of why).
- **Long-running jobs on the pod**: always launch with `nohup ... > log 2>&1 < /dev/null &` + `disown` — the SSH connection to this pod has had a few transient timeouts (unrelated to the training process itself), so anything multi-hour must survive a dropped connection.

## Known gotchas (read before editing notebooks)

1. **Part 2's notebook stores cell `source` as a single string**, not a list of lines (Part 3's notebooks use the standard list format). Any scripted find/replace across cells MUST handle both — join to a string, edit, then reconstruct the original type. Getting this wrong silently corrupts the file into a per-character array (still valid JSON, `''.join()` reads back fine, but the raw file bloats ~8x and byte-level substrings break). Always verify with `ast.parse()` per cell after patching, and check file size stays sane before committing.
2. **`num_workers` must be 0 on Windows** when running the notebook-converted `.py` script locally — the script isn't guarded with `if __name__ == "__main__":`, so `DataLoader(num_workers>0)` triggers Windows to respawn and re-execute the whole script, crashing with a multiprocessing bootstrap `RuntimeError`. The committed notebooks use `num_workers=4` (correct for RunPod/Linux); only the local-laptop-only patched copy (`part2/part2_run.py`, gitignored) overrides it back to 0.
3. **The "package outputs" zip step self-includes**: `shutil.make_archive(..., WORKING_DIR)` zips the entire working directory, including any zip from a previous run sitting inside it — reruns compound and can fill a disk (this happened once, 166GB, fixed by deleting stale zips before archiving). Fixed in all 9 notebooks as of 2026-08-08; the fix is a `glob.glob(*.zip) + os.remove()` cleanup right before `make_archive`. Preserve this if regenerating any of these cells.
4. **Python 3.14** (the system default `python`) is too new for current PyTorch wheels — always use the `a:/thesis/.venv` (Python 3.11) for anything touching torch.
5. Console `print()`/`grep` on these notebooks can hit `UnicodeEncodeError`/false-negative issues (emoji in markdown cells, compact-JSON files read as few giant "lines") — prefer writing diagnostic output to a file and reading it back, and use occurrence-counting (`text.count(...)`) rather than `grep -c` (which counts matching *lines*, misleading on compact JSON).
6. **The water290 dataset (`faizalkarim/flood-area-segmentation`) contains at least one mislabeled file**: `Image/0.jpg` is actually a GIF (header `GIF89a...`), not a JPEG. `cv2.imread` returns `None` on it and crashes the next line; PIL's `Image.open` handles it fine regardless of extension. Part 1's water290 visualization cell was fixed to use PIL instead of cv2 (2026-08-09, applied on the pod first — check it's been backported to `part1/fed-salgate-v2-part-1.ipynb` in git, see PROGRESS.md). The actual training data path already used PIL and was never affected.
7. **Notebook-converted scripts on RunPod: always launch with `nohup`+`disown`**, never a plain foreground/backgrounded SSH command — the pod's SSH connection has had transient timeouts unrelated to the training process itself, and a bare `&` inside a single SSH command can still die with the session on some setups.
8. When patching a script after a partial run crashed partway through, check whether the crashed step already wrote real output to disk (e.g., mask generation) before blindly restarting from scratch — add a skip-if-already-done guard rather than re-paying an expensive step. (Applied to Part 1's mask-generation step on RunPod, saved ~35 min on the restart.)
9. **RunPod stop/resume only persists `/workspace`** — everything else in the container filesystem, including packages installed by hand mid-session (not part of the script's own `pip install` list or the base `runpod/pytorch` image), resets to the base image on every resume. Hit this 2026-08-10: `pandas`/`seaborn` had been manually installed earlier and vanished after a stop/resume cycle, crashing a script at `import pandas`. Always `pip install` anything manually added last session before relying on it again post-resume.
10. **Secure Cloud pods with no network volume (pod-local disk only) can fail to resume** with `"not enough free GPUs on the host machine"` if that specific physical host is fully occupied by other renters — this pod (`0lrs9f0awbogvn`) has `networkVolume: null`, so it's tied to one host machine and can't just be recreated elsewhere. This is a wait-and-retry situation (worked on the 4th attempt, seconds apart), not a lost-pod situation — don't assume data loss.
11. **Python default arguments are bound at `def`-time, not call-time.** `saliency_consensus_gates(client_models, probe_loader, beta=GATE_BETA, tau=GATE_TAU)` looked like it would honor a later `GATE_BETA`/`GATE_TAU` reassignment, but the defaults were frozen the moment the `def` executed — reassigning the globals afterward silently had zero effect at the call site (`run_federated` called it with no explicit args). Any script that needs to vary a "default-bound" global across a loop (e.g. a hyperparameter sweep) must patch the call site to pass the values explicitly (`beta=GATE_BETA, tau=GATE_TAU` at the call, not just as defaults in the `def`), so the lookup happens dynamically at call time.
