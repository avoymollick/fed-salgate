# Progress Log

Dated record of what was actually run, what results came out, and what's next. Updated after every significant milestone (experiment run, bug fix, decision). See `CLAUDE.md` for stable project context.

---

## 2026-07-31 — Part 1 executed on Kaggle (by Navin)

Ran `part1/fed-salgate-v2-part-1.ipynb` in full on Kaggle (Tesla T4). Real results, not estimates:

- Heuristic (hand-coded color-rule) pseudo-masks vs real GT: **mIoU 0.052** (confirms the warning in the notebook's own comments — treat as noise).
- Self-trained DeepLabV3+ (ResNet-34, 4-channel with SPADANet-style water-prior channel), 40 epochs on merged FloodNet-GT + water290 pool: **held-out mIoU 0.382**, pixel accuracy 0.818.
- Dirichlet non-IID partition (α=1.5, 3 clients): sizes **852 / 333 / 201**, all Grass-dominant (mild non-IID).
- Fixed two real bugs found during this run: (a) mask-filename matching only matched exact stems, silently dropping all real GT masks (`_lab` suffix mismatch) — fixed with suffix-tolerant matching; (b) water-detection heuristic required blue MNDWI AND blue HSV together, missing brown/turbid flood water — fixed to OR + morphological cleanup.
- **Note**: val mIoU was still climbing (0.158→0.343 through epoch 24) when training stopped at 40 — hadn't clearly plateaued. Candidate for re-run with more epochs (see Open Decisions below).
- Outputs (masks/, checkpoint, config.json, client_indices.pkl, gt_val_indices.pkl) live in Navin's Kaggle kernel output, not yet on RunPod.

## 2026-08-07 — Repo setup, data staging, local demo run

- Initialized git repo at `a:/thesis`, organized into `part1/part2/part3/docs`, pushed to `git@github.com:avoymollick/fed-salgate.git` (private).
- Generated dedicated SSH keypairs for GitHub and RunPod (`~/.ssh/github_fedsalgate`, `~/.ssh/runpod_fedsalgate`).
- Staged data locally for laptop testing: FloodNet (24GB, verified counts match Kaggle: 51/1047/450/448), water290 (107MB, 290 images), and Part 1's outputs pulled via `kaggle kernels output navinxqz/fed-salgate-v2-part-1` (had to retry — first two attempts silently stalled from a duplicate-process collision; fixed by killing stale processes and retrying clean).
- Built local Python 3.11 venv (`a:/thesis/.venv`) with torch 2.5.1+cu121 — Python 3.14 (system default) is too new for current PyTorch wheels.
- Converted `part2/notebook3896666d14.ipynb` to a script, patched for local Windows paths + a fast demo config (`NUM_ROUNDS=6, BATCH_SIZE=2, CLIENT_SUBSAMPLE=32`), fixed a Windows-specific `DataLoader(num_workers>0)` multiprocessing crash (set `num_workers=0` for the local-only copy).
- **First local demo run result** (RTX 4050 laptop, 6-round demo scale — NOT final numbers):

  | Model | Held-out mIoU | Pixel Acc |
  |---|---|---|
  | DeepLabV3+ / FedAvg | 0.2586 | 0.7794 |
  | DeepLabV3+ / FedProx | 0.2720 | 0.7892 |
  | **DeepLabV3+ / Fed-SalGate** | **0.2729** | **0.7940** |
  | LiteUNet / FedAvg | 0.1301 | 0.4938 |
  | LiteUNet / Fed-SalGate | 0.1545 | 0.5034 |

  Fed-SalGate beat both baselines (+0.0143 vs FedAvg, +0.0009 vs FedProx). Published as a shareable report artifact for the group.

## 2026-08-08 — Speedups for RunPod: num_workers/cudnn, then AMP, then a real bug found+fixed

**Pass 1 (safe, mechanical):** `num_workers: 2→4`, `cudnn.benchmark: False→True` across all 8 model notebooks. First attempt silently corrupted Part 2's notebook (8x file bloat, edit didn't even apply) — its `source` field is a plain string, not a list-of-lines like Part 3's notebooks; a naive `for line in src` iterated characters instead of lines. Caught before committing, reverted, rewrote the patch to be type-safe. Verified byte-level on all 8 files before committing. Pushed.

**Pass 2 (AMP — automatic mixed precision):** Added `torch.cuda.amp.autocast()` + `GradScaler` to all three distinct training-loop variants (Part 2's federated `_local_train`, Part 3.07 FedSeg's `_local_train`, and the shared `train_centralized` used by the 6 centralized Part 3 baselines), plus eval-time autocast wherever `torch.no_grad()` appears. Patch script self-validates: exact-match assertion per block + `ast.parse()` per cell before saving, aborts with nothing written on any mismatch.

Verified with an actual end-to-end local run (not just syntax-checked):

  | Model | Held-out mIoU | Pixel Acc |
  |---|---|---|
  | DeepLabV3+ / FedAvg | 0.2596 | 0.7884 |
  | DeepLabV3+ / FedProx | 0.2779 | 0.7987 |
  | DeepLabV3+ / Fed-SalGate | 0.2694 | 0.7867 |
  | LiteUNet / FedAvg | 0.1274 | 0.4905 |
  | LiteUNet / Fed-SalGate | 0.1544 | 0.5046 |

  AMP training is numerically stable (sensible loss/mIoU, no NaN). **Note**: Fed-SalGate did *not* beat FedProx this run (−0.0085) — expected noise at this tiny demo scale (6 rounds, 32-image subsample), not a regression; don't read into which method "won" at demo scale, only that the pipeline runs correctly. No local wall-clock speedup observed, because the laptop is forced to `num_workers=0` (Windows bug, see CLAUDE.md gotcha #2) — CPU-bound data loading is masking AMP's compute-side win locally. Should show up properly on RunPod once `num_workers=4` is actually active.

**Bug found during this verification run**: the "package outputs" zip step zips its own working directory into itself, so reruns compound — this run's zip alone grew to **166GB** and filled the 300GB drive to 0 bytes free before erroring (`OSError: No space left on device`) mid-archive. Training itself completed fine; only the zip step failed. Fixed across all 9 notebooks (model notebooks + merge notebook) by deleting old zips before creating the new one. Deleted the corrupted 166GB file, recovered disk space. Committed and pushed both fixes together.

---

## Open decisions (not yet acted on)

- **Part 1 re-run**: bump `SELFTRAIN_EPOCHS` 40→100 with early stopping (patience~15 on held-out mIoU) — val mIoU hadn't plateaued at epoch 40. High leverage since Part 1's pseudo-masks feed everything downstream.
- **Non-IID severity**: current 3-client/α=1.5 split is mild (all Grass-dominant). Considering 5 clients/α=0.3 for a more dramatic non-IID story before the real Part 2 run — would require re-running Part 1's partition cell (cheap, doesn't need re-training) and would invalidate the currently-pulled local `client_indices.pkl`.
- **RunPod**: on hold per user request while laptop testing continues. SSH key ready, needs: pod provisioned (user's call — costs money), pubkey added, pod SSH string shared back.
- **Budget estimate given to user**: ~$6-8 (Community Cloud RTX 4090) to ~$17 worst case (Secure Cloud, full ablations, 1.5x buffer), well under the $20 the group set aside.
- **HuggingFace**: token not yet provided; needed for staging Part 1's large artifacts (masks/checkpoint) to RunPod without routing through the local laptop, per the 3-leg transfer plan (GitHub=code, Kaggle=source datasets, HF=derived artifacts).
