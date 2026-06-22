# RAG + Dense Retrieval System

> 🚧 In progress — started June 2026

End-to-end retrieval-augmented QA system built from scratch.
Reproduces the core mechanisms of DPR and RAG, extended with a cross-encoder re-ranking ablation.

---

## Papers Reproduced

| Paper | Authors | Year |
|-------|---------|------|
| Dense Passage Retrieval (DPR) | Karpukhin et al. | 2020 |
| Retrieval-Augmented Generation (RAG) | Lewis et al. | 2020 |
| Self-RAG | Asai et al. | 2023 |

---

## What This Builds

**Phase 1 — Dense Retriever (DPR-style)**
- Dual-encoder: separate question encoder + passage encoder (fine-tuned from BERT)
- In-batch negative sampling for contrastive training — implemented from scratch
- FAISS index over passage embeddings, approximate nearest-neighbour retrieval
- Evaluation: Recall@1, Recall@5, Recall@10

**Phase 2 — Generator (RAG-style)**
- Fusion-in-Decoder conditioning: retrieve top-k passages, fuse with query, generate with T5
- RAG-sequence vs RAG-token marginalization comparison

**Phase 3 — Extension**
- Cross-encoder re-ranking stage added after retrieval
- Ablation: BM25 vs dense vs hybrid (BM25 + dense) first-stage retrieval
- Measure how much re-ranker compensates for weaker first-stage retrieval

**Phase 4 — Write-up**
- Short paper-style report: method, ablation table, error analysis
- Dataset: Natural Questions subset (+ domain-specific eval TBD)

---

## Results (to be filled)

| System | Recall@10 | Exact Match |
|--------|-----------|-------------|
| BM25 baseline | — | — |
| DPR (reproduced) | — | — |
| DPR + Cross-encoder re-ranker | — | — |
| Hybrid BM25+Dense + Re-ranker | — | — |

---

## Structure

```
rag-retrieval-system/
├── retriever/
│   ├── dual_encoder.py       # Question + passage encoders
│   ├── train_retriever.py    # In-batch negative training loop
│   └── faiss_index.py        # Index build + ANN retrieval
├── generator/
│   ├── fusion_in_decoder.py  # FiD-style conditioning
│   └── rag_sequence.py       # RAG-sequence marginalization
├── reranker/
│   └── cross_encoder.py      # Cross-encoder re-ranking stage
├── eval/
│   └── metrics.py            # Recall@k, Exact Match, F1
├── notebooks/
│   ├── 01_retriever_training.ipynb
│   ├── 02_generator_training.ipynb
│   └── 03_ablation_study.ipynb
├── report/
│   └── writeup.md            # Paper-style write-up
└── README.md
```

---

## Key Implementation Decisions

- In-batch negatives: why this works and where it breaks
- FAISS index type choice (Flat vs IVF vs HNSW) and the speed/recall tradeoff
- Why Fusion-in-Decoder outperforms simple concatenation
- What Self-RAG's reflection tokens add and the training cost

---

## Part of

[`nlp-from-scratch`](https://github.com/Tanveer-heir/nlp-from-scratch) curriculum ·
[Tanveer-heir](https://github.com/Tanveer-heir)
