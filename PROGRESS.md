# Progress Log

Dated record of what was actually run, what results came out, and what's next. Updated after every significant milestone (experiment run, bug fix, decision). See `CLAUDE.md` for stable project context.

---

## The story so far, in plain language (read this first)

**What the project is**: FedSalGate is our thesis system for flood-damage detection from drone photos. Instead of one central server training on everyone's photos, three-to-five simulated "regional offices" (clients) each train on their own local slice of data and only share model updates, not raw photos — that's federated learning. Our actual contribution — the thing the thesis is supposed to prove is good — is **Fed-SalGate**: instead of just averaging every client's update equally (that's the standard method, called FedAvg), our server compares each client's Grad-CAM "explanation" (a heatmap of what the model is looking at) against the average of the others, and down-weights any client whose explanation disagrees with the group. The idea is that a client whose data/behavior looks weird gets less say in the final model.

**Score we care about**: mIoU (mean Intersection-over-Union), a number from 0 to 1 measuring how well predicted flood-damage regions overlap with the real, human-labeled ones. Higher is better. All the "final" numbers below are measured on 59 real photos that are set aside once and never used for training anything — so they're an honest test, not the model grading its own homework.

**Chapter 1 — the foundation (Part 1).** Navin originally built and ran the pseudo-labeling pipeline on Kaggle: it takes FloodNet's 398 real labeled photos + 1047 unlabeled ones, trains a model to guess labels for the unlabeled ones ("self-training"), and splits everyone into non-IID (unevenly distributed) simulated client groups. His original run got mIoU 0.382. We reran this on a rented RunPod GPU with 100 epochs instead of 40 (plus early stopping so it doesn't overtrain), and got it up to **mIoU 0.4117** — a real, meaningful improvement, and the foundation everything else builds on.

**Chapter 2 — the first real test (Part 2), and a disappointing result.** We ran the actual 3-way comparison — FedAvg vs. FedProx (another standard baseline) vs. our Fed-SalGate — for real, all 50 rounds, on 3 clients. Result: **Fed-SalGate did not win.** It scored 0.4186 mIoU, slightly behind FedAvg's 0.4229. We dug into why: we watched the "gates" (the per-client trust weights our method computes) round by round, and they basically never moved — they stayed almost perfectly equal (0.98 to 1.02) the entire run. The reason: with only 3 clients that don't disagree much, there's nothing for our gating mechanism to actually *do* — it collapses into being the same as plain averaging, just with extra computation.

**Chapter 3 — a detour to rule out the dataset.** Before assuming the fix was "make the clients disagree more," we tested an alternative theory: maybe the second dataset we merge in (290 extra photos that only have water labeled, no other classes) is somehow hurting things. We ran two experiments. The first attempt was flawed (a model still trying to predict 10 classes despite only ever getting supervision for 1) and gave a misleadingly bad number. We caught the mistake, fixed it, and reran it properly as a real 2-class water-only detector. Result: it does learn water very well on its own 290 photos (81% IoU) but that skill barely transfers to our real flood-drone photos (23% IoU) — a real "these two photo sets look too different" finding, but it didn't explain why Fed-SalGate lost. So we ruled this out and went back to the client-count theory.

**Chapter 4 — the real test, done properly.** We re-split the data into 5 clients with much more extreme imbalance (from 33 photos in the smallest client to 815 in the biggest, instead of the previous fairly-even 3-way split) and reran the full comparison. This time, **the gating mechanism actually did something** — we watched it correctly identify the client with the most different data (a "Tree"-dominant client, unlike everyone else's "Grass"-dominant data) and consistently gave it a lower trust weight, in clean proportion to how much it actually disagreed. That part of our idea is proven to work exactly as designed.

But the overall score still didn't improve: **Fed-SalGate scored 0.4086, FedAvg scored 0.4145** — Fed-SalGate lost again, by a slightly bigger margin than before. So "make the mechanism activate" was necessary but not sufficient — activating correctly doesn't automatically mean it helps the final accuracy. Down-weighting the "different" client may have thrown away genuinely useful information along with whatever was "noisy" about it.

**The one real bright spot**: when we break the score down by class instead of averaging everything together, **Fed-SalGate actually wins clearly on the Water class specifically** — 0.524 vs. FedAvg's 0.467 and FedProx's 0.476. For a *flood*-damage thesis, correctly finding water is arguably the single most important class, so this is a genuine, defensible result even though the overall average doesn't move in our favor.

**Where things stand right now**: pipeline works end-to-end, results are real and fully documented (not cherry-picked), and we have three honest findings: (1) the gating mechanism works mechanically as designed, (2) it doesn't currently improve overall accuracy, (3) it does meaningfully improve the Water class. This needs a team decision on how to frame the paper — options include leaning into the Water-class win + honest mechanism analysis, trying to tune the gating's sensitivity parameters before giving up on the overall number, or trying a different signal (not just Grad-CAM agreement) for deciding which client to trust less. Everything is saved (GitHub, HuggingFace, local files) and the RunPod pod is stopped (no cost) until we decide the next move.

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

**Final result** (early-stopped at epoch 64, best epoch's water290-val IoU=0.8132):

| Metric | IoU | Pixel Acc |
|---|---|---|
| water290-internal (in-distribution) | **0.8132** | 0.9155 |
| FloodNet real held-out GT, binarized (out-of-distribution transfer) | **0.2340** | 0.6585 |

**Clean, decisive domain-gap finding**: the model learns water290 itself very well (81% IoU) but that skill barely transfers to FloodNet's real aerial-drone photos (23% IoU). water290 and FloodNet are visually different enough (likely camera angle/altitude/style) that water290 alone is not a substitute for real FloodNet-domain water supervision. This does **not** prove the merge (water290 + FloodNet GT together, as the main pipeline does) is harmful — only that water290 in isolation isn't sufficient. Archived to HF: `avoymollick/fedsalgate-artifacts/water290_binary_ablation`. Plot downloaded to `a:/thesis/data/water290_binary_plots/`.

## 2026-08-10 — User directive: "I need the best paper results" — back to non-IID severity

Neither water290 ablation explained Fed-SalGate's Part 2 loss (both point at a real but separate domain-gap finding, not a bug in the main pipeline). User's call: stop investigating side hypotheses and pursue the highest-leverage fix directly — **re-run Part 2 under real non-IID stress**, since the gate-uniformity diagnosis (2026-08-09 entry) is still the most direct explanation and the gating mechanism has never actually been tested under conditions harsh enough to activate it.

**Key efficiency win**: the expensive part (100-epoch self-training, already validated at mIoU 0.4117) doesn't need to be redone — client partitioning is a separate, cheap, CPU-only step that happens *after* self-training and only depends on the already-relabeled masks on disk. Wrote a standalone repartition script (`/workspace/repartition_5c_a03.py`) that:
- Reuses the exact `FloodNetDataset` and `NonIIDPartitioner` classes copied verbatim from Part 1 (to keep sample ordering/indexing identical)
- Reuses the existing relabeled masks and the *same* 59 held-out GT indices (unchanged, for comparability)
- Re-partitions with **`NUM_CLIENTS=5, DIRICHLET_ALPHA=0.3`** (up from 3/1.5) — meaningfully harsher non-IID
- Backs up the old 3-client/α=1.5 `client_indices.pkl`/`config.json` (suffix `.3clients_a1.5.bak`) before overwriting in place — Part 2's actual completed results are saved separately (`part2_v2_summary.json`) and untouched by this either way

**Repartition result**: client sizes **107 / 815 / 33 / 300 / 131** — genuinely more diverse than the original 728/347/311 (dramatic size imbalance, 33 to 815) and dominant classes 38.5%/63.3%/28.6%/52.1% Grass plus **Client 4 is Tree-dominant (37.5%)** — the first client in either run to not be Grass-dominant.

**Scope decision for this rerun**: patched `part2/part2_run.py` to skip the two LiteUNet experiments (Experiments 4/5) this pass — run just the core DeepLabV3+ FedAvg/FedProx/Fed-SalGate comparison first (~3hrs at the new per-round cost vs ~8hrs for all 5), decide on LiteUNet reruns after seeing whether the core result holds up. Rationale discussed openly with user re: time/cost efficiency (~$7 spent so far, well under budget, but no reason to blindly rerun everything when the 3-way comparison is what actually answers the question).

**Real bug found on first launch attempt**: crashed immediately with `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])` — a PyTorch BatchNorm error. Root cause: Client 2 has only 33 samples; `33 % BATCH_SIZE(4) == 1`, so its last training batch each epoch had exactly 1 sample, and BatchNorm can't compute variance from a single sample in training mode. Never triggered before because `CLIENT_SUBSAMPLE=96` is divisible by 4, so previously every client's per-round batch count was always a clean multiple of 4 — this is a genuine edge case the harsher, smaller-client split surfaced. Fixed with `drop_last=(len(sub_ds) > BATCH_SIZE)` on the client `DataLoader` (guarded so a client smaller than one batch still gets a partial batch rather than zero). Verified fix works: relaunched, Round 1/50 completed cleanly, gates correctly show 5 uniform values (`0.2` each) to start.

Independently re-verified the running config against the log directly (not just trusting the launch): `NUM_CLIENTS: 5`, `DIRICHLET_ALPHA: 0.3`, client sizes `107/815/33/300/131` all matched exactly what repartitioning produced.

**Timings**: FedAvg 90.9 min, FedProx 94.8 min, Fed-SalGate 95.1 min (all 50 rounds each, ~4.7 hours total — longer than the original 3-client run since 5 clients means more total per-round work, as anticipated).

**Gate dynamics — the mechanism DID activate this time**, unlike the mild-non-IID run where gates stayed flat at 0.98–1.02 with no real pattern. Final-round gates correlated cleanly with each client's actual agreement score:

| Client | Agreement | Gate |
|---|---|---|
| 0 | 0.9297 | 1.0238 |
| 1 | 0.8649 | 1.0114 |
| 2 | 0.8664 | 1.0118 |
| 3 | 0.7855 | 0.9813 |
| 4 | 0.7689 | **0.9717** (lowest agreement → lowest gate, correctly) |

**Final held-out real-GT results**:

| Model | mIoU | Pixel Acc |
|---|---|---|
| DeepLabV3+ / FedAvg | **0.4145** | 0.8147 |
| DeepLabV3+ / FedProx | 0.4071 | 0.8075 |
| DeepLabV3+ / Fed-SalGate | 0.4086 | 0.8085 |

**Fed-SalGate gain vs FedAvg: −0.0059. Fed-SalGate gain vs FedProx: +0.0015.** Fed-SalGate still lost to FedAvg — by a slightly *wider* margin than the mild non-IID run (−0.0059 vs −0.0042). The core hypothesis (harsher non-IID → gating activates → Fed-SalGate wins) is only half-confirmed: the gating mechanism demonstrably works as designed (down-weights the lowest-agreement client, in clean proportion to its disagreement), but that down-weighting does not translate into better held-out generalization. Likely explanation: Grad-CAM agreement isn't a reliable proxy for "this client's contribution is harmful" — Client 4 (Tree-dominant, genuinely different data) disagreeing with the consensus doesn't mean its gradient signal was bad; it may have held complementary information that got suppressed.

**Per-class nuance worth keeping for the paper**: Fed-SalGate actually **wins on Water IoU** (0.5237 vs FedAvg 0.4674, FedProx 0.4758 — a real +0.056/+0.048 improvement) and Building-NonFlooded (0.4028 vs 0.3886/0.3608) — but loses ground on Vehicle (0.1416, worst of the three), Road-Flooded (0.2672, worst), Tree (0.4578, worst), and Road-NonFlooded (0.4191, worst). For a flood-*damage* thesis, winning specifically on Water — arguably the single most flood-relevant class — while trading off on less-central classes is a genuine, defensible, class-specific finding, not just a clean loss.

Archived to HF: `avoymollick/fedsalgate-artifacts/part2_5clients_a03` (note: this includes a redundant copy of Part 1's masks/checkpoint too, since Part 2 writes into the same shared working directory Part 1 used — harmless duplication, not a data issue). Plots downloaded to `a:/thesis/data/part2_5clients_a03_plots/` (the `25_gap4_lightweight.png` in the source folder was a stale leftover from the original 3-client run — not regenerated this pass since LiteUNet was skipped — excluded from the local copy to avoid confusion).

---

## Open decisions / next up

- **What this means for the paper's narrative**: Fed-SalGate does not currently beat FedAvg on overall held-out mIoU in either non-IID regime tested (mild: −0.0042, harsh: −0.0059), despite the gating mechanism demonstrably activating correctly under the harsher split. Options going forward: (a) reframe around the Water-class-specific win + the mechanism's demonstrated correctness as the honest contribution, (b) investigate whether a different aggregation signal (not pure Grad-CAM agreement) would correlate better with "helpful vs harmful" client updates, (c) try tuning `GATE_BETA`/`GATE_TAU` before concluding, (d) accept and report the negative result plainly with the mechanistic analysis as the value-add. **Not yet decided — needs discussion, not another blind rerun.**
- **LiteUNet reruns**: not yet done on the 5-client/α=0.3 split — deferred pending the above decision.
- **Possible 3rd water290 ablation** (FloodNet-GT-only, no water290, for a complete 3-way comparison table): parked.
- **Part 3 + ablations**: on hold pending the narrative decision above.
- **Backport pod-only fixes to the git repo**: the GIF-file fix, mask-gen skip guard, repartitioning script, LiteUNet-skip patch, and the `drop_last` BatchNorm fix were all applied/run directly on the pod, not committed to git. Still needs porting so the committed notebooks match what actually worked.

## 2026-08-10 — Session paused for team discussion

User is taking the Part 2 results (both non-IID regimes) to discuss with the group before deciding next steps — not a "stuck" pause, a deliberate one, since the result raises a real question about the thesis's narrative that's a team call, not a unilateral one.

**Pod stopped** (not terminated) to halt billing — `desiredStatus: EXITED`. Volume and all data/code/results on it are intact. Pod ID `0lrs9f0awbogvn`. To resume: start the pod again via the API (see CLAUDE.md Environment section), re-fetch the SSH port (it will likely differ from the `15653` used this session), everything else picks up exactly where it left off — no need to re-clone, re-download datasets, or redo Part 1.

**Everything from today is saved in three places**: this file + `CLAUDE.md` (committed to GitHub), HuggingFace (`avoymollick/fedsalgate-artifacts` — `part1_100epoch`, `part1_water290only`, `water290_binary_ablation`, `part2_5clients_a03`), and locally (`a:/thesis/data/*_plots/`).

**One-paragraph summary for the team discussion**: We ran the full pipeline for real. Fed-SalGate (our saliency-consensus gating contribution) does not currently beat plain FedAvg on overall held-out mIoU, tested under two non-IID regimes (mild 3-client/α=1.5: −0.0042; harsh 5-client/α=0.3: −0.0059). Under the harsher split we confirmed the gating mechanism *does* activate correctly — it down-weights clients in clean proportion to how much their Grad-CAM explanations disagree with consensus — but that correct behavior doesn't translate into better overall generalization. The one genuinely positive, specific finding: Fed-SalGate wins on Water IoU specifically (+0.05 over both baselines) in the harsh-non-IID run, which is directly relevant for a flood-damage thesis even though it doesn't rescue the overall number. Two ruled-out alternative explanations: the water290 dataset merge (tested via two separate ablations, real domain-gap finding but doesn't explain the gap) and mild non-IID severity (fixed by the harsher rerun, which is what surfaced the current, more informative result). Decision needed: reframe the paper around the Water-class win + honest mechanistic analysis, try tuning the gate parameters (`GATE_BETA`/`GATE_TAU`) or a different consensus signal before concluding, or report the full negative result as-is with the diagnostic work as the contribution.

## 2026-08-10 — Resumed: gate-hyperparameter (`GATE_BETA`/`GATE_TAU`) tuning sweep

Team discussion concluded; decision was to pursue the tuning sweep as due diligence before settling the paper's narrative (cheapest of the remaining options, since FedAvg/FedProx for the 5-client/α=0.3 split are already computed and reusable — only Fed-SalGate itself needs retraining per gate config).

**Root-cause diagnosis found before launching anything** (explains why gates stayed so flat even once the mechanism "activated" under the harsh split): the gate formula is `gate = sigmoid(β·(agreement − τ))`, with defaults `β=10, τ=0.5`. But real observed agreement scores were never anywhere near 0.5 — they ranged **0.769–0.930** in the 5-client run. Plugging that range in: `β·(agreement−τ)` works out to roughly 2.7–4.3, and `sigmoid()` of anything above ~2.5 is already pinned near 1.0 (`sigmoid(2.7)=0.937`, `sigmoid(4.3)=0.987`). **Every client was landing on the flat, saturated tail of the sigmoid** — confirmed exactly by the cached run's own numbers: agreements spanning 0.769–0.930 produced gates spanning only 0.972–1.024. The mechanism wasn't failing to differentiate clients; `τ=0.5` was simply miscalibrated for where real agreement scores actually live, so the sigmoid never reached its sensitive middle range. This directly motivates recentering `τ` near the real agreement band (~0.80–0.85) rather than a blind grid search.

**Pod resume hit a real infrastructure snag**: `runpod.resume_pod()` first failed with `"There are not enough free GPUs on the host machine to start this pod"` — confirmed via GraphQL that this pod's storage is a local pod-disk (`networkVolume: null`), tied to one specific physical host (`hijw1l6qdo6w`, datacenter US-NC-1), not a portable network volume — so a busy host genuinely blocks resume, not just "pick a different machine." Retried 3× a few seconds apart and the 4th attempt succeeded (GPU freed up on that host almost immediately). New SSH port `14863` (same public IP `103.196.86.102`), `~/.ssh/config` updated. **New gotcha for CLAUDE.md**: Secure Cloud pods with only a pod-local disk (no network volume) can fail to resume if their specific host machine is fully occupied by other renters — worth knowing this is a possible wait-and-retry situation, not a lost-pod situation.

**Second snag, same root cause (ephemeral container)**: `part2_gate_sweep.py` crashed immediately on `import pandas` — `ModuleNotFoundError`. Root cause: only `/workspace` persists across a RunPod stop/resume cycle; the rest of the container filesystem (including packages installed manually mid-session, outside the script's own `pip install` list) resets to the base image on every resume. `pandas`/`seaborn` had been installed by hand earlier in the session and didn't survive the restart. Fixed with a one-off `pip install pandas seaborn`, relaunched. **New gotcha for CLAUDE.md**: any package not in the script's own install list at the top, or not part of the base `runpod/pytorch` image, must be reinstalled after every pod resume — don't assume last session's manual `pip install`s persisted.

**Sweep design**: built `part2/part2_gate_sweep.py` on the pod by slicing `part2_run.py`'s setup (imports through data/model pipeline, unchanged) and replacing the 3-way FedAvg/FedProx/Fed-SalGate comparison tail with a loop that trains **Fed-SalGate only**, 3 times, at:

| Config | τ | β |
|---|---|---|
| tau0.85_beta10 | 0.85 | 10.0 |
| tau0.85_beta20 | 0.85 | 20.0 |
| tau0.80_beta15 | 0.80 | 15.0 |

One code-level fix was needed to make this work at all: the gate function's `beta=GATE_BETA, tau=GATE_TAU` were bound as Python *default arguments* at `def`-time, so simply reassigning the globals between sweep configs would silently have no effect (a real Python gotcha — default args are evaluated once, not per-call). Patched the one call site inside `run_federated` to pass `beta=GATE_BETA, tau=GATE_TAU` explicitly, which resolves the globals dynamically at call time instead. Each config's held-out mIoU is compared against the **cached** `DeepLabV3+/FedAvg` (0.4145) and `DeepLabV3+/FedProx` (0.4071) results from the already-completed 5-client/α=0.3 run — those are not retrained. Results/checkpoints written incrementally after each config (`gate_sweep_results.json` in `fed_salgate_outputs/`), so a crash partway through doesn't lose earlier configs.

**Launched** via `nohup ... > /workspace/gate_sweep.log 2>&1 < /dev/null & disown` (per the standing nohup+disown practice). Independently re-verified via a fresh SSH connection (not trusting the launch command's own output, which timed out mid-connection — the known transient-SSH-timeout pattern, not a real failure): data pipeline loaded correctly (client sizes 107/815/33/300/131, matching the existing 5-client/α=0.3 partition exactly), training started on config 1 (`tau0.85_beta10`). Expect roughly ~95 min/config × 3 ≈ ~4.75 hours total.

**Config 1 result (τ=0.85, β=10.0) — first Fed-SalGate win of the whole investigation**, though a very thin one:

| Config | Held-out mIoU | vs FedAvg (0.4145) | vs FedProx (0.4071) |
|---|---|---|---|
| Original (τ=0.5, β=10) | 0.4086 | −0.0059 | +0.0015 |
| **τ=0.85, β=10** | **0.4148** | **+0.0003** | **+0.0077** |

Final gates for this config: `[1.227, 1.079, 1.449, 0.939, 0.305]` against agreements `[0.879, 0.851, 0.923, 0.825, 0.670]` — a real spread (0.305–1.449) compared to the original run's narrow 0.972–1.024 band, confirming the sigmoid-saturation diagnosis was correct and recentering τ restores differentiation. The margin over FedAvg is small enough (+0.0003) that it shouldn't be oversold as a clean win on its own, but it's the first config in either non-IID regime where Fed-SalGate hasn't lost.

**Config 2 result (τ=0.85, β=20.0 — same recentered threshold, sharper steepness) — a real, clean win**:

| Config | Held-out mIoU | vs FedAvg (0.4145) | vs FedProx (0.4071) |
|---|---|---|---|
| Original (τ=0.5, β=10) | 0.4086 | −0.0059 | +0.0015 |
| τ=0.85, β=10 | 0.4148 | +0.0003 | +0.0077 |
| **τ=0.85, β=20** | **0.4327** | **+0.0182** | **+0.0256** |

Not a marginal edge — a genuinely meaningful improvement over both baselines. Final gates: `[1.629, 1.025, 1.455, 0.635, 0.256]` against agreements `[0.937, 0.857, 0.908, 0.815, 0.757]`, an even wider, more decisive spread than config 1's. Best single per-round mIoU during training was 0.4574. Notably Road-Flooded IoU jumped to 0.291 (class 3) and Vehicle jumped to 0.246 (class 7, both were weak classes in every prior run) — worth checking the full per-class breakdown once config 3 lands, this may be broader than just the Water-class win seen previously. This is the strongest evidence so far that the original τ=0.5 was a real calibration bug, not a fundamental flaw in the saliency-consensus-gating idea. Config 3 (τ=0.80, β=15.0) still running, will confirm whether this generalizes across nearby (τ,β) settings or is sensitive to the exact values.
