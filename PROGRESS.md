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

Also created `CLAUDE.md` and `PROGRESS.md` (this file) to track project context and run history going forward.

## 2026-08-09 — RunPod deployed, pod bootstrapped, Part 1 re-run launched

User gave full delegated access: RunPod API key (deploy pods directly, not just SSH in after manual creation) and a HuggingFace write token.

- **RunPod pod deployed via API** (`runpod` Python SDK, not the web UI): RTX 4090, 24GB VRAM. Community Cloud had no stock available at request time, fell back to **Secure Cloud ($0.74/hr** vs $0.34/hr Community — still within the $20 budget, matches the earlier "worst case" estimate). First deploy attempt didn't inject the SSH key correctly (`PUBLIC_KEY=null` in pod env, connection refused) — terminated that pod (0 uptime, no cost) and redeployed passing the key explicitly via `env={"PUBLIC_KEY": ...}`. SSH confirmed working: `ssh runpod` alias configured locally.
- **Pod bootstrapped**: cloned the GitHub repo (copied the dedicated `github_fedsalgate` deploy key to the pod — it's a purpose-built key for this repo, not the user's main credentials), copied `kaggle.json` over, installed torch (2.4.1+cu124, already present in the RunPod PyTorch image) + segmentation-models-pytorch/albumentations/etc.
- **Datasets pulled directly on the pod** (not through the laptop): FloodNet at 245-320MB/s (vs ~4-8MB/s on the home laptop connection — finished in under 2 minutes) and water290. All counts verified exactly matching Kaggle: 51/347/1047/450/448 FloodNet, 290 water290.
- **HuggingFace**: created private dataset repo `avoymollick/fedsalgate-artifacts` for archiving generated results/checkpoints (backup + easy sharing with groupmates, independent of pod lifecycle). Clarified with user: HF is an **export destination** for what we generate, not something RunPod needs to pull from to get started — the ImageNet backbone auto-downloads from `download.pytorch.org`, and Part 1→Part 2 handoff happens locally on the same pod disk.
- **User correction on scope**: keep Navin's original Part 1 config as-is (3 clients, α=1.5, both datasets merged as already validated) — do **not** change the non-IID split this round. Only change: epochs 40→100 with early stopping. Plan: run Part 1, then Part 2, then reassess (possibly revisit the binary-water-masking merge logic) before Part 3/ablations.
- **Part 1 patched for RunPod** (`part1/fed-salgate-v2-part-1.py`, converted from the notebook via nbconvert directly on the pod): hardcoded `IN_KAGGLE=False` + direct `/workspace/data/...` paths (previous local-Windows path-patching lesson applied: used small anchor-based text inserts instead of matching a large multi-line block, since the pod's nbconvert output had different incidental whitespace than expected — large-block exact-match failed once, caught immediately via the assertion, no bad write). `SELFTRAIN_EPOCHS` 40→100, added early stopping (patience=15 on held-out val mIoU). `NUM_CLIENTS=3`/`DIRICHLET_ALPHA=1.5` confirmed unchanged. Syntax-validated (`ast.parse`) before writing.
- **Part 1 launched** on the pod (`nohup ... &`, detached from the SSH session so it survives the connection drops we've been seeing — RunPod's SSH has had a couple of transient timeouts, unrelated to the training process itself, which stays alive independently). Mask generation actually sped up partway through and finished in **34.9 min** (not the ~100 min estimated).
- **Real crash found and fixed**: right after mask generation, hit `cv2.error: ... !_src.empty() in function 'cvtColor'` in the water290 "visualize merged samples" cell. Root cause: `Image/0.jpg` in the `faizalkarim/flood-area-segmentation` Kaggle dataset is **actually a GIF file with a `.jpg` extension** (confirmed via file header: `GIF89a...`) — a genuine data-quality issue in the source dataset, not something introduced by the RunPod download. `cv2.imread` can't read GIF and silently returns `None`, then the next line crashes on it. This only affects that one visualization cell — the real training path loads images via PIL (`Image.open`), which handles GIF transparently regardless of extension, so training data itself was never at risk. Fixed by switching that cell to load via PIL instead of cv2.
- **Also added**: a skip-if-already-generated guard around the mask-generation step, so restarting the script after a downstream crash doesn't redo the already-completed 35-minute mask generation. Restarted — confirmed the skip worked (back to where it crashed in under 2 minutes instead of 35).
- Both fixes applied directly on the pod via small Python patch scripts (anchor-based text replace + `ast.parse` validation before writing, same safety pattern established during the local AMP patching work) — **not yet ported back to the notebook in the git repo**. Need to backport `part1/fed-salgate-v2-part-1.ipynb` with the same two fixes once Part 1 finishes successfully, so the committed notebook matches what actually worked.
- **Self-training result (100 epochs, RTX 4090)**: ran the full 100 epochs, early stopping never triggered (val mIoU kept improving enough to stay under the 15-epoch patience threshold throughout — LR decayed via `ReduceLROnPlateau` at epochs 30/43/58/74/86/94, roughly halving each time down to 1.56e-06 by epoch 100). Loss dropped steadily 1.65→0.125. **Final held-out val mIoU: 0.4117** (best, epoch 88) vs the original 40-epoch run's 0.382 — a real **+0.0297 (~7.8% relative) improvement**, confirming the "more epochs" call was worth it. Epoch-1 numbers (loss=1.647, mIoU=0.159) matched the original Kaggle run almost exactly, confirming the RunPod environment is behaving consistently with the original setup, not producing a different result for spurious reasons.
- **Part 1 fully complete** — no errors, all 6 plots generated (`00`–`04`, `10`), masks written (1047 unlabeled / 450 val / 448 test), checkpoint saved. New Dirichlet partition (same seed, α=1.5, 3 clients) came out **728 / 347 / 311** — noticeably different from the original run's 852/333/201, because the better self-trained model produces different pseudo-labels, which shifts the per-image dominant-class computation feeding the partition. Not a bug — expected consequence of retraining. All 3 clients still Grass-dominant (47.6–59.9%).
- **Archived to HuggingFace**: full `/workspace/working` (295MB — masks, checkpoint, config, client_indices.pkl, gt_val_indices.pkl, all plots) pushed to `avoymollick/fedsalgate-artifacts/part1_100epoch`. Verified file listing after upload (all 11 plots, incl. the 5 "plain-language proof pack" plots that generate after the "PART 1 COMPLETE" banner prints). This is now the durable copy — survives pod termination.
- Downloaded all 11 Part 1 plots to the laptop too (`a:/thesis/data/part1_plots_100epoch/`).

## 2026-08-09/10 — Part 2 full run (50 rounds, 3 clients/α=1.5): Fed-SalGate did NOT beat FedAvg

Ran Part 2 in full on the pod (converted notebook → script, only 2 lines needed patching — `IN_KAGGLE=False` and `WORKING_DIR='/workspace/working'` — since Part 2 shares Part 1's working dir by design and `FLOODNET_BASE` was already correct in Part 1's saved `config.json`). All speedups from 2026-08-08 (AMP, `num_workers=4`, `cudnn.benchmark`, zip fix) were already baked into the committed notebook, no extra patching needed for those.

**Round-log mIoU (pseudo-val subset, during training)**:

| Model | Round mIoU (best) | Wall time |
|---|---|---|
| DeepLabV3+ / FedAvg | 0.4607 | 62.9 min |
| DeepLabV3+ / FedProx | 0.4582 | 62.9 min |
| DeepLabV3+ / Fed-SalGate | 0.4629 | 66.8 min |
| LiteUNet / FedAvg | 0.2242 | 56.7 min |
| LiteUNet / Fed-SalGate | 0.1677 | 59.4 min |

**Final held-out real-GT mIoU (the number that actually matters)**:

| Model | mIoU | Pixel Acc |
|---|---|---|
| DeepLabV3+ / FedAvg | **0.4229** | 0.8124 |
| DeepLabV3+ / FedProx | 0.4174 | 0.8064 |
| DeepLabV3+ / Fed-SalGate | 0.4186 | 0.8124 |
| LiteUNet / FedAvg | **0.2682** | 0.7616 |
| LiteUNet / Fed-SalGate | 0.2262 | 0.7681 |

**Fed-SalGate gain vs FedAvg: −0.0042. Fed-SalGate gain vs FedProx: +0.0012.** Fed-SalGate lost narrowly to FedAvg on DeepLabV3+ (won on the round-log/pseudo-val metric but lost on the held-out one) and lost more clearly on LiteUNet.

**Diagnosis**: watched the saliency-consensus gates throughout every round checked — they stayed within ~0.98–1.02 of uniform (1.0) the *entire* 50 rounds, never meaningfully deviating. The gating mechanism only does something when clients' Grad-CAM explanations disagree; with the current 3-client/α=1.5 split (mild non-IID — all 3 clients Grass-dominant, 47.6–59.9%), clients apparently agree too much for gating to ever activate. Fed-SalGate effectively collapsed to FedAvg-plus-overhead. This matches the pre-planned trigger condition for revisiting non-IID severity.

Total pod time for Part 2: ~5 hours across the 5 experiments.

User's call after seeing this: rather than pursue the non-IID severity fix immediately, revisit the water290 binary-masking dataset first, as an explicit ablation (user's framing: "we will again check part 1 and part 2 with binary masking only as experiment"). Non-IID severity stays on the list, deferred.

## 2026-08-09 — Ablation 1: self-train on water290 ONLY (flawed methodology, caught and corrected)

**What was run**: a separate copy of Part 1 (`part1/part1_water290only.py`, output to its own `/workspace/working_water290only` directory so the good merged-training results were never touched), with the self-training data pool changed to *only* the 290 water290 images — FloodNet's 398 real GT images excluded from training entirely (kept only for the validation split, for a comparable held-out number). Also fixed a latent bug this surfaced: the per-class loss-weight formula (`1/(freq+eps)`) assigns huge phantom weights to classes with zero training pixels; with only water290 in the pool, 9 of 10 classes have zero pixels, so the naive formula would've crushed water's own normalized weight to ~0.004. Fixed to zero-out absent classes instead.

**Result**: training loss collapsed almost instantly (1.036→0.009, full memorization of the tiny 290-image set); held-out **10-class** mIoU never moved off ~0.02 across 19 epochs (early-stopped, best 0.0228) — worse than even the original heuristic baseline (0.052).

**User caught a real methodology flaw in this experiment**: the model architecture was left at `NUM_CLASSES=10` (10 output channels) even though training data only ever supervised one class (Water) via `ignore_index=255` on everything else. The other 9 channels never received a gradient update, so at eval time the model was outputting near-random logits for 9 classes and getting scored against them anyway — conflating "how good is water290 at teaching water detection" with "what happens when you starve 9 unrelated output channels." The 0.0228 number is real (nothing fabricated) but doesn't cleanly answer the actual question. Archived to HF anyway (`avoymollick/fedsalgate-artifacts/part1_water290only`) as a documented negative result / cautionary methodology note, not as the real answer.

## 2026-08-09/10 — Ablation 2: proper binary (2-class) water detector — the corrected version

Built as a **standalone, focused script** (`/workspace/water290_binary.py`, not a patched copy of Part 1's pipeline) specifically to answer the real question cleanly:
- Genuine 2-class model: `DeepLabV3Plus(classes=2)`, water vs. not-water, plain `CrossEntropyLoss` (no `ignore_index` needed — "not water" is now a real, supervised class)
- Trained/early-stopped (patience=15, ReduceLROnPlateau) on an 80/20 split of water290 itself — this is the in-distribution number
- **The number that actually matters**: final evaluation on the *same 59 FloodNet held-out real photos* used throughout Part 1/2 (loaded via the same `gt_val_indices.pkl`, ground truth binarized: class 5=Water → 1, else → 0) — this is the out-of-distribution transfer number, i.e. does water290 alone actually help on real flood-drone photos.
- Sanity-checked the held-out-index reconstruction logic *before* the full run (59 indices, 51+347=398 total matching config exactly, sample mask correctly contains class 5) rather than discovering a bug after an hour of training.

In progress as of this log entry: epoch 17/100, water290-internal val IoU = 0.7814 (78%, pixel acc 90%) — strong in-distribution result, as expected for a properly-scoped binary task. Final FloodNet-transfer IoU still pending.

---

## Open decisions / next up

- **Binary water detector transfer result**: pending — this is what actually tells us whether water290 is worth keeping in the merge at all, or whether it's not pulling its weight.
- **Non-IID severity (5 clients/α=0.3)**: still the other live hypothesis for why Fed-SalGate didn't beat FedAvg in the Part 2 run — deferred behind the binary-masking investigation per user's explicit call, not abandoned.
- **Part 3 + ablations**: on hold until both of the above are resolved.
- **Backport pod-only fixes to the git repo**: the GIF-file fix and the mask-gen skip guard (Part 1) were applied directly on the pod's converted `.py` script, not to `part1/fed-salgate-v2-part-1.ipynb` in git. Still needs porting into the notebook so the committed version matches what actually worked. Same applies to any further Part 1 changes made along the way.
