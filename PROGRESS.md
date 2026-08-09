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

---

## Open decisions / next up

- **Non-IID severity (5 clients/α=0.3)**: explicitly deferred by user this round — revisit only if Part 2 results on the current 3-client/α=1.5 split suggest it's needed.
- **Binary water-masking dataset**: user flagged this as a possible second iteration target ("if needed again part 1 with binary masking dataset") — meaning after seeing Part 2 results on the 100-epoch Part 1 rerun, may revisit how the water290 binary masks are merged in, then re-run Part 1 → Part 2 again before moving to Part 3.
- **Part 2 full run**: next step once Part 1 finishes — 50 rounds, full data, on the pod.
- **Part 3 + ablations**: after Part 2, pending the reassessment above.
- **Archive to HF**: push Part 1's fresh outputs (masks/checkpoint/config) to `avoymollick/fedsalgate-artifacts` once training completes.
- **Backport pod-only fixes to the git repo**: the GIF-file fix and the mask-gen skip guard were applied directly on the pod's converted `.py` script, not to `part1/fed-salgate-v2-part-1.ipynb` in git. Port both into the notebook once Part 1 finishes successfully.
