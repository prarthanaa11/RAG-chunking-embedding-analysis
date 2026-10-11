
# Comparative Analysis of Chunking Strategies and Embedding Models on RAG Retrieval


---

## 1. Project Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with generative AI to provide responses grounded in external documents. The quality of a RAG system depends significantly on how documents are divided into chunks, how those chunks are represented as embeddings, and how relevant information is retrieved for a given query.

This project investigates the effect of different **chunking strategies, chunk sizes, overlap settings, and embedding models** on the quality and efficiency of retrieval in a RAG pipeline.

The experiments will use technical documentation as the primary dataset and compare retrieval configurations using quantitative evaluation metrics. The findings will help identify suitable configurations for building reliable and efficient RAG systems.

## 2. Problem Statement

RAG systems frequently rely on default document chunking and embedding configurations without systematically evaluating whether these choices are appropriate for a particular dataset or retrieval task.

Fixed-size chunking may split related information across chunks, while sentence-level and semantic chunking may preserve context differently. Similarly, embedding models may vary in their ability to retrieve relevant passages from technical documents.

There is a need for a systematic comparative analysis of chunking strategies and embedding models to understand their impact on retrieval relevance, context coverage, computational cost, and retrieval latency.

## 3. Objectives

- Compare fixed-size, sentence-level, and semantic chunking strategies.
- Investigate the effect of different chunk sizes and overlap settings.
- Evaluate multiple embedding models, including suitable open-source models and an API-based model if resources permit.
- Implement vector-based retrieval using FAISS and/or Chroma.
- Measure retrieval quality using precision, recall, and other suitable retrieval metrics.
- Evaluate retrieval latency and computational requirements.
- Develop a comparative decision matrix recommending configurations for different technical-document retrieval scenarios.

## 4. Research Questions

1. How does the choice of chunking strategy affect retrieval quality?
2. How do chunk size and overlap influence the relevance and completeness of retrieved context?
3. How do different embedding models compare when retrieving information from technical documentation?
4. What trade-offs exist between retrieval quality, latency, and computational cost?
5. Which chunking and embedding configurations are most suitable for different RAG use cases?

## 5. Scope

The project focuses on the **retrieval component of a RAG pipeline**, with an emphasis on controlled experiments.

The scope includes:

- Technical-document collection and preprocessing.
- Fixed-size, sentence-level, and semantic chunking.
- Experiments with different chunk sizes and overlap values.
- Embedding generation using selected models.
- Vector indexing and similarity-based retrieval.
- Evaluation using a predefined set of queries and relevant-document labels.
- Comparative analysis, visualizations, and recommendations.

The project does not initially aim to train a new embedding model or develop a production-scale RAG application. Generative answer evaluation may be included as an extension after the retrieval experiments are established.

## 6. Proposed Methodology

The project will follow these stages:

1. **Dataset preparation:** Collect suitable technical documents and prepare a representative query set.
2. **Preprocessing:** Clean documents while preserving meaningful textual structure.
3. **Chunking:** Generate chunks using fixed-size, sentence-level, and semantic approaches.
4. **Embedding generation:** Convert chunks into vector representations using selected embedding models.
5. **Vector indexing:** Store embeddings in FAISS or Chroma for similarity-based retrieval.
6. **Query retrieval:** Retrieve the top-k relevant chunks for each evaluation query.
7. **Evaluation:** Measure retrieval quality, latency, and computational requirements.
8. **Comparison:** Analyze the results and identify effective configurations.
9. **Recommendation:** Prepare a decision matrix based on the observed experimental results.

### Planned experimental variables

| Variable | Planned configurations |
|---|---|
| Chunking strategy | Fixed-size, sentence-level, semantic |
| Chunk size | Multiple predefined sizes |
| Chunk overlap | Zero, low, and moderate overlap |
| Embedding model | Multiple selected embedding models |
| Vector store | FAISS and/or Chroma |
| Retrieval parameter | Different top-k values |
| Evaluation | Retrieval quality, latency, resource usage |

The final values will be selected after the initial baseline and literature review. All comparisons will use consistent evaluation queries and document collections wherever possible.

## 7. Evaluation Metrics

The following metrics will be considered:

- **Precision@k:** Proportion of the top-k retrieved results that are relevant.
- **Recall@k:** Proportion of all known relevant results retrieved within the top-k results.
- **MRR (Mean Reciprocal Rank):** Measures how highly the first relevant result appears in the ranked results.
- **Context relevance:** Measures whether the retrieved passages are relevant to the query.
- **Context recall:** Measures how much of the required relevant information is retrieved.
- **Retrieval latency:** Time required to retrieve results for a query.
- **Computational requirements:** Memory usage, embedding-generation time, and other resource costs where measurable.

If a generative model is incorporated, groundedness or faithfulness can also be evaluated separately. Retrieval metrics and answer-generation metrics will not be treated as interchangeable.

## 8. Initial Baseline

The Week 1 baseline establishes a simple working retrieval pipeline before introducing additional experimental configurations.

The initial implementation uses:

- Sample technical documents.
- Fixed-size, word-based chunking.
- A predefined chunk-overlap value.
- The `sentence-transformers/all-MiniLM-L6-v2` embedding model.
- FAISS for vector similarity search.
- Normalized embeddings and inner-product similarity.
- Top-k retrieval for user queries.

This baseline serves as a reference point for subsequent experiments. It is not the final comparative result.

## 9. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Development environment | Visual Studio Code |
| Embedding models | Sentence Transformers and selected alternatives |
| Vector search | FAISS and/or Chroma |
| Data processing | NumPy, Pandas |
| Experimentation | Jupyter Notebook |
| Version control | Git and GitHub |
| Evaluation | Python-based retrieval metrics and result analysis |

Open-source models will be prioritized for reproducibility. Any API-based model will be evaluated only if access, cost, and project constraints permit.

## 10. Repository Structure

```text
rag-chunking-embedding-analysis/
│
├── data/
│   ├── raw/                  # Original technical documents
│   └── processed/            # Cleaned or prepared data
│
├── notebooks/                # Exploratory analysis and experiments
│
├── src/
│   ├── baseline.py           # Initial retrieval pipeline
│   ├── chunking.py           # Chunking strategies
│   ├── embeddings.py         # Embedding model integration
│   ├── retrieval.py          # Vector indexing and retrieval
│   └── evaluation.py         # Evaluation metrics
│
├── evaluation/
│   └── queries.csv           # Queries and relevance information
│
├── literature_review/
│   └── papers_summary.csv    # Research-paper comparison
│
├── results/                  # Experimental results and figures
│
├── requirements.txt
├── .gitignore
└── README.md
```

The structure represents the planned organization. Modules will be added as the corresponding features are implemented.

## 11. Installation and Setup

### Prerequisites

- Python 3.11 recommended for the initial environment.
- Git.
- Visual Studio Code.
- A GitHub account.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/rag-chunking-embedding-analysis.git
cd rag-chunking-embedding-analysis
```

Replace `YOUR-USERNAME` with the GitHub username that owns the repository.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Run the baseline

```bash
python src/baseline.py
```

The initial baseline accepts a query and displays the highest-ranked retrieved chunks, their source documents, similarity scores, and retrieval time.

The first run may require downloading the embedding model.

## 12. Team Responsibilities

All four members will contribute to the literature review, project discussions, evaluation design, and review presentations. The following assignments identify primary responsibilities.

| Member | Primary responsibility |
|---|---|
| Member 1 | Literature survey, research gap, and problem formulation |
| Member 2 | Dataset preparation, preprocessing, and chunking strategies |
| Member 3 | Embedding models, vector indexing, and retrieval implementation |
| Member 4 | Evaluation, experiment integration, result visualization, and documentation |

Responsibilities may overlap as required for integration, testing, and final analysis. Each member will document their contributions through GitHub commits, issues, and pull requests.

## 13. Project Plan

### Week 1 – Problem Definition and Baseline

- Finalize the research problem and objectives.
- Review relevant research papers.
- Establish the GitHub repository and development environment.
- Prepare initial technical documents and evaluation queries.
- Implement and test a fixed-size chunking baseline.

### Subsequent Weeks – Implementation and Evaluation

- Expand the dataset and evaluation queries.
- Implement additional chunking strategies.
- Integrate and compare embedding models.
- Run controlled retrieval experiments.
- Calculate evaluation metrics and record latency.
- Analyze results and prepare visualizations.
- Develop the decision matrix and complete the IEEE-style report.

The detailed schedule will be refined according to faculty review requirements and the team's implementation progress.

## 14. Expected Deliverables

- A documented literature review and research gap.
- A working baseline retrieval pipeline.
- Implementations of the selected chunking strategies.
- Experiments comparing embedding models.
- A reproducible evaluation dataset and query set.
- Quantitative experiment results and visualizations.
- A comparative decision matrix with evidence-based recommendations.
- A final IEEE-format project report and review presentation.

## 15. Reproducibility and Limitations

To support fair comparisons, experiments will record the dataset version, chunking configuration, embedding model, retrieval parameters, evaluation queries, and relevant runtime conditions.

Results may depend on the document collection, query distribution, model version, hardware, and implementation choices. Conclusions will therefore be limited to the tested configurations and documented experimental conditions.

## 16. References

1. P. Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” *Advances in Neural Information Processing Systems*, 2020. [Paper](https://arxiv.org/abs/2005.11401)
2. N. Reimers and I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” *Proceedings of EMNLP-IJCNLP*, 2019. [Paper](https://arxiv.org/abs/1908.10084)
3. N. Thakur et al., “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models,” 2021. [Paper](https://arxiv.org/abs/2104.08663)
4. N. Muennighoff et al., “MTEB: Massive Text Embedding Benchmark,” *Proceedings of EACL*, 2023. [Paper](https://arxiv.org/abs/2210.07316)
5. G. Izacard et al., “Unsupervised Dense Information Retrieval with Contrastive Learning,” 2021. [Paper](https://arxiv.org/abs/2112.09118)

Additional references on semantic chunking, chunk size, and RAG evaluation will be incorporated as the literature review progresses.

---

**Project status:** Week 1 — Initial setup, literature review, and baseline implementation.  
**Note:** Experimental comparisons and recommendations will be reported only after the relevant experiments have been conducted.
