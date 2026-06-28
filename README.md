# Dense Retrieval + RAG: DPR & RAG Reproduction with Hybrid Retrieval and Hard-Negative Analysis

A from-scratch reproduction of **DPR** (Karpukhin et al., 2020) and **RAG** (Lewis et al., 2020), trained and evaluated on two corpora with different lexical characteristics, extended with **hybrid BM25 + dense retrieval** and a controlled **hard-negative mining experiment** broken down by answer-type category.

## Overview

This project implements and evaluates a full retrieval-augmented question answering pipeline across two independently-trained setups:

1. **Retriever**: Dual-encoder models (`bert-base-uncased`) trained per-corpus with in-batch negative contrastive loss.
2. **Hybrid retrieval**: Combines the trained dense retriever with BM25 (sparse, keyword-based) retrieval via score normalization and weighted blending.
3. **Hard-negative retrieval (controlled experiment)**: A second retriever per corpus, trained with an explicit BM25-mined hard negative added to every training batch, to test whether harder negative sampling closes the dense-vs-sparse performance gap.
4. **Category-level analysis**: Validation questions are categorized by gold-answer type (numeric / date / proper noun / other), and every retrieval method is compared within each category — not just in aggregate.
5. **Generator**: A `t5-base` model generating answers from retrieved passages, in a simplified Fusion-in-Decoder (FiD) style.

**The central finding**: dense vs. sparse retrieval performance, and the effect of hard-negative training, are both **highly corpus-dependent**. The pattern observed on SQuAD does not transfer to TriviaQA — in some cases it reverses entirely. This cross-corpus contrast, not a single clean "X beats Y" result, is the project's main empirical contribution.

## Datasets

| | SQuAD | TriviaQA |
|---|---|---|
| Source | `rajpurkar/squad` | `mandarjoshi/trivia_qa` (rc.wikipedia) |
| Raw examples sampled | 87,000 | 40,000 |
| Unique passages (corpus) | 18,890 | 21,111 |
| Validation questions | 1,500 | 1,000 |
| Passage style | Short, curated Wikipedia paragraphs (avg. ~120 words) | Longer, less curated Wikipedia evidence documents (truncated to 200 words) |

TriviaQA was deliberately chosen as a second corpus with messier, more lexically varied text than SQuAD's clean curated paragraphs, to test whether retrieval method comparisons generalize across corpus styles.

## Architecture

### Retriever (DPR)
- Two independent `bert-base-uncased` encoders per corpus (question encoder, passage encoder)
- Embedding = `[CLS]` token's final hidden state (768-dim), L2-normalized before similarity computation
- **In-batch negative loss**: for each question in a batch, its true passage is the positive; every other passage in the batch is a free negative
- Trained for 3 epochs, batch size 16 (32 effective with dual-GPU), learning rate 2e-5, temperature 0.1
- Per-epoch checkpointing, with automatic resume-from-last-checkpoint on interrupted runs

### Hard-negative retriever (controlled experiment)
- A second, separately-initialized retriever per corpus
- For each training question, BM25 mines the top-10 lexically similar passages; the highest-ranked one that is **not** the gold passage is used as an explicit hard negative
- Each training step computes loss over `[B, 2B]` scores (B in-batch positives/negatives + B hard negatives), rather than the baseline's `[B, B]`
- Trained from scratch with the same hyperparameters as the baseline, isolating the effect of negative-sampling strategy

### Retrieval index
- FAISS `IndexFlatIP` (exact inner-product search) per corpus, per retriever variant (baseline and hard-negative each get their own index)
- **Hybrid retrieval**: top-100 candidates from dense and BM25 independently, scores min-max normalized, combined as `alpha * dense + (1-alpha) * bm25`

### Generator (RAG / simplified FiD)
- `t5-base`, fed `"question: {q} context: {p1} context: {p2} ..."` as a single concatenated sequence
- Beam search decoding (`num_beams=4`)
- Simplified single-sequence approximation of FiD — the original paper encodes each passage separately and fuses representations before decoding; this implementation concatenates instead, which is simpler but scales less gracefully beyond ~3-5 passages

### Answer-type categorization
Validation questions are bucketed by their gold answer using transparent heuristics (not a trained NER model):
- **date**: contains a 4-digit year or month name
- **numeric**: answer is mostly digits/currency/percentage
- **proper_noun**: every word is capitalized
- **other**: everything else

This is an approximation, not a gold-standard linguistic categorization — some edge cases will misclassify (e.g., a capitalized common noun at a sentence boundary).

## Results

### Retrieval: Recall@k — Dense vs. Hard-Negative-Dense vs. Hybrid (α=0.5)

**SQuAD**

| k | Dense | Hard-Neg Dense | Hybrid |
|---|---|---|---|
| 1 | 37.7% | 38.9% | **58.7%** |
| 5 | 63.8% | 64.9% | **78.5%** |
| 20 | 81.3% | 81.4% | **89.7%** |
| 100 | 93.5% | 93.6% | **95.4%** |

**TriviaQA**

| k | Dense | Hard-Neg Dense | Hybrid |
|---|---|---|---|
| 1 | 57.4% | **64.9%** | 51.4% |
| 5 | 76.7% | **79.7%** | 72.7% |
| 20 | 86.0% | **87.8%** | 87.1% |
| 100 | 94.0% | 94.0% | 94.0% |

**This is the key cross-corpus contrast.** On SQuAD, hybrid retrieval dominates at every k, and hard-negative training has almost no effect on dense retrieval. On TriviaQA, the pattern partially reverses: hard-negative dense retrieval is the *best* method at R@1 and R@5, beating both the baseline dense retriever and the α=0.5 hybrid blend.

### Alpha ablation (blend weight sweep)

**SQuAD** — optimum at α=0.5

| α | R@1 | R@5 | R@20 | R@100 |
|---|---|---|---|---|
| 0.0 (pure BM25) | 49.3% | 66.4% | 76.9% | 85.8% |
| 0.3 | 57.9% | 75.8% | 84.0% | 86.4% |
| **0.5** | **58.7%** | **78.5%** | **89.7%** | **95.4%** |
| 0.7 | 57.2% | 78.6% | 88.4% | 93.5% |
| 1.0 (pure dense) | 37.7% | 63.8% | 81.3% | 93.5% |

**TriviaQA** — optimum shifts toward pure dense, α=0.5 underperforms pure dense

| α | R@1 | R@5 | R@20 | R@100 |
|---|---|---|---|---|
| 0.0 (pure BM25) | 33.2% | 49.3% | 64.0% | 75.2% |
| 0.3 | 46.5% | 67.2% | 73.4% | 75.8% |
| 0.5 | 51.4% | 72.7% | 87.1% | 94.0% |
| 0.7 | 55.8% | 77.2% | 88.1% | 94.0% |
| **1.0 (pure dense)** | **57.4%** | 76.7% | 86.0% | 94.0% |

On SQuAD, BM25 alone (49.3% R@1) is a strong standalone baseline and blending helps substantially. On TriviaQA, BM25 alone is comparatively weak (33.2% R@1), and the best-performing configuration is dense-heavy, not a balanced blend — the opposite tendency from SQuAD. A fixed α=0.5 (the value validated on SQuAD) is **not transferable** to TriviaQA, where it actively underperforms pure dense retrieval at R@1.

### Hard-negative training: effect on generation quality (200-sample EM/F1)

| Corpus | Metric | Dense | Hard-Neg | Hybrid |
|---|---|---|---|---|
| SQuAD | EM | 43.5% | 43.0% | **55.0%** |
| SQuAD | F1 | 51.6% | 50.9% | **64.1%** |
| TriviaQA | EM | 30.5% | 29.5% | **35.5%** |
| TriviaQA | F1 | 37.1% | 36.4% | **43.9%** |

At the generation level (combining retrieval with end-to-end QA accuracy), hybrid retrieval is the strongest configuration on **both** corpora — even though hard-negative dense retrieval edged out hybrid on raw TriviaQA Recall@1/5. This suggests hard-negative dense retrieval's gains at the top of the ranking don't fully carry through to the passages that matter most for the generator's final answer quality, where hybrid's broader recall advantage still wins out.

### Recall@5 by answer category

**SQuAD**

| Category (n) | Dense | Hard-Neg | BM25 | Hybrid |
|---|---|---|---|---|
| proper_noun (440) | 67.0% | 66.6% | 64.8% | **77.3%** |
| numeric (81) | 74.1% | 69.1% | 76.5% | **87.7%** |
| date (103) | 64.1% | 67.0% | 64.1% | **77.7%** |
| other (876) | 61.2% | 63.4% | 66.6% | **78.3%** |

**TriviaQA**

| Category (n) | Dense | Hard-Neg | BM25 | Hybrid |
|---|---|---|---|---|
| numeric (7) | 85.7% | 85.7% | 71.4% | 71.4% |
| proper_noun (723) | 77.5% | **78.8%** | 48.8% | 72.2% |
| date (12) | 66.7% | 75.0% | 58.3% | **100%** |
| other (258) | 74.8% | **82.2%** | 49.6% | 72.9% |

**Sample-size caveat**: TriviaQA's `numeric` (n=7) and `date` (n=12) categories are far too small to support reliable conclusions — a single example flipping changes the percentage by 8-14 points. These rows are included for completeness but should not be treated as evidence of a real effect. SQuAD's categories (81-876 examples) are large enough to be more trustworthy.

Two patterns hold up at reasonable sample sizes:
- On **SQuAD**, hybrid wins in every category, including the larger `proper_noun` (n=440) and `other` (n=876) buckets — the hybrid advantage is broad, not concentrated in one answer type.
- On **TriviaQA**'s largest category (`proper_noun`, n=723) and `other` (n=258), **hard-negative dense retrieval outperforms both BM25 and hybrid** — BM25 is notably weak here (48-50%), dragging hybrid down with it. This is consistent with the alpha ablation finding that TriviaQA favors dense-leaning configurations.

### Qualitative example (original SQuAD experiment, smaller-scale run)

> **Q**: How many Lucille Lortel Awards was *In Transit* nominated for?
> **Gold**: four
> **Dense-fed generation**: eleven *(incorrect)*
> **Hybrid-fed generation**: four *(correct)*

A case where BM25's exact-match strength recovered a passage the dense retriever missed, directly correcting a generation error.

## Discussion

The headline result is **not** "hybrid retrieval is universally better" or "hard negatives universally help" — it is that **the relative strength of dense, sparse, and hybrid retrieval is corpus-dependent, and a fusion weight or training strategy tuned on one corpus does not reliably transfer to another.**

Plausible explanations, offered as hypotheses rather than confirmed conclusions:
- **TriviaQA's passages are longer and less curated** than SQuAD's, which may dilute BM25's term-matching signal (more irrelevant text per passage competing for TF-IDF weight) while giving the dense encoder more context to extract a useful semantic signature.
- **Hard-negative mining via BM25's top-1-wrong passage** may be a noisier negative signal on TriviaQA, where passages are longer and a "lexically similar but wrong" passage might still share substantial topical overlap with the gold passage, weakening the contrastive signal — yet this same mechanism appears to help TriviaQA's dense retriever's R@1/R@5 while not helping SQuAD's at all, so this explanation is incomplete and worth further investigation.
- **SQuAD's curated, single-paragraph contexts** may make exact term matches unusually reliable (questions were authored by people looking directly at the paragraph), inflating BM25's relative strength on this corpus in a way that may not generalize to naturally-occurring text.

## Engineering notes

- **Loss collapse to exactly 0**: initial training saw loss crash to `0.0000` within ~150 steps. Root cause: un-normalized `[CLS]` embeddings produced unbounded dot products, amplified further by a small temperature (0.05), saturating the softmax almost immediately. Fixed via L2-normalization before similarity computation and adjusting temperature to 0.1.
- **CUDA OOM**: batch sizes of 64 and 16 exceeded a single T4's 16GB VRAM during early runs. Resolved via reduced batch size (8, later 16 with dual-GPU), trimmed sequence lengths, and explicit memory cleanup between steps.
- **Multi-GPU (T4 x2)**: hard-negative retraining used `nn.DataParallel` across both GPUs, roughly halving wall-clock training time relative to a single-GPU equivalent (estimated ~3 hours actual vs. ~5-6 hours estimated single-GPU for SQuAD's hard-negative run).
- **Kaggle session persistence**: `/kaggle/working/` output does not persist automatically between sessions — only a committed Version with verified output survives, and a "Save & Run All" commit always re-executes from a clean container rather than inheriting a live session's in-progress state. A live training session's progress was lost mid-project when this was misunderstood; recovered by locating an earlier committed version's output, attaching it as an input source, and copying files into the working directory. All subsequent expensive steps (training, encoding, indexing, evaluation) were given skip-if-exists checks against `/kaggle/working/`, and checkpoints were downloaded locally after each major step as an independent safety net.

## Limitations & honest caveats

- Hard-negative mining uses a simple heuristic (BM25 top-1-excluding-gold) rather than the more sophisticated iterative/dense-embedding-based hard-negative mining used in some DPR follow-up work; results may differ with a stronger mining strategy.
- The FiD implementation is a simplified single-sequence concatenation, not the original paper's per-passage encode-then-fuse architecture.
- Generation evaluation (EM/F1) was run on 200-example subsets per corpus for compute efficiency, not the full validation sets.
- Answer-type categorization is heuristic (regex/capitalization-based), not a trained NER system, and some categories (especially in TriviaQA) have sample sizes too small for reliable per-category conclusions.
- The cross-corpus contrast is based on two datasets; whether the observed corpus-dependence generalizes further (e.g., to fully open-domain Natural Questions with a full Wikipedia index) is untested here due to the infrastructure cost of the original DPR paper's 21M-passage corpus.

## Tech stack

`PyTorch` · `Transformers (HuggingFace)` · `FAISS` · `rank-bm25` · `datasets` · Dual NVIDIA T4 GPUs (Kaggle)

## Papers reproduced

- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.*
- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.*
