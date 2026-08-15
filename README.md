# Multimodal Document Intelligence RAG

An end-to-end **multimodal Retrieval-Augmented Generation (RAG)**
pipeline for answering questions over PDF documents containing both
textual and visual information such as **charts, tables, diagrams,
screenshots, maps, and scanned pages**.

Unlike a text-only RAG pipeline, this project preserves document visuals
during ingestion and re-attaches the original page images during answer
generation. The system produces **page-level citations**, exposes a
retrieval trace, and includes an evaluation loop for measuring retrieval
quality, groundedness, correctness, and citation precision.

## Overview

Traditional text-only RAG can lose important information when the answer
exists only inside a chart, table, diagram, or scanned page.

This project uses a two-stage multimodal approach:

1.  **Extract the PDF text layer and render every page as an image.**
2.  **Use a vision-capable model to understand each page's visual
    content.**
3.  **Convert visual information into structured textual
    representations** that can be embedded alongside normal text.
4.  **Create text and visual chunks** in a unified retrieval space.
5.  **Retrieve using both dense embeddings and BM25 sparse retrieval.**
6.  **Fuse the rankings using Reciprocal Rank Fusion (RRF).**
7.  **Re-attach the original page images for the highest-ranked pages.**
8.  **Generate a grounded answer with structured page-level citations.**
9.  **Evaluate retrieval and generation independently.**

The key design principle is:

> **Search over a unified text representation, but generate from both
> retrieved text and the original page images.**

This makes the retrieval layer relatively simple and debuggable while
allowing the generation model to verify visual information directly from
the source page.

## Architecture

``` text
                         ┌──────────────────────┐
                         │      PDF Document    │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            Extract Text Layer              Render Page Images
                    │                               │
                    │                               ▼
                    │                    ┌─────────────────────┐
                    │                    │ Vision Model        │
                    │                    │ Charts / Tables /   │
                    │                    │ Diagrams / Scans   │
                    │                    └──────────┬──────────┘
                    │                               │
                    ▼                               ▼
             Text Chunks                    Visual Chunks
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Embedding Model     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                 ┌────────────────────────────────────┐
                 │ Unified Dense + Sparse Index       │
                 │                                    │
                 │ Dense: NumPy / FAISS               │
                 │ Sparse: BM25                       │
                 └────────────────┬───────────────────┘
                                  │
                              User Query
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              Dense Retrieval            BM25 Retrieval
                     │                         │
                     └────────────┬────────────┘
                                  ▼
                         RRF Rank Fusion
                                  │
                                  ▼
                         Top-K Retrieved Chunks
                                  │
                                  ▼
                    Original Page Images Added
                                  │
                                  ▼
                         Multimodal LLM
                                  │
                                  ▼
                   ┌──────────────────────────┐
                   │ Grounded Answer          │
                   │ + Page Citations        │
                   │ + Visual Evidence Flag  │
                   └──────────────────────────┘
```

## Why Multimodal RAG?

A text-only RAG pipeline may successfully retrieve the surrounding
paragraph while still missing the information contained in a figure.

For example, a PDF page may contain:

-   A bar chart with numerical values
-   A table whose values are not available in the text layer
-   A system architecture diagram
-   A scanned document
-   A screenshot
-   A map
-   An equation or other visual information

The project therefore creates a **visual textual surrogate** at indexing
time. The vision model extracts structured information such as:

-   Figure type
-   Figure caption
-   Figure description
-   Axis labels and units
-   Legible numerical values
-   Table contents
-   Section heading
-   Named entities
-   Whether information is visual-only

The original page image is still retained so the generation model can
verify the source rather than relying only on the generated description.

## Project Components

### 1. PDF ingestion

Each PDF page produces a `PageAsset` containing:

-   `doc_id`
-   `page_no`
-   Extracted text
-   Base64-encoded page image
-   Image dimensions
-   `text_poor` flag

Pages with little or no embedded text are identified as potentially
scanned or graphical pages.

The default configuration uses:

-   Render DPI: `150`
-   Maximum image edge: `1536 px`
-   Text-poor threshold: `180` characters

### 2. Vision-based page understanding

Each page is sent to the vision model using a structured Pydantic
schema.

The extraction contains:

``` text
PageExtraction
├── summary
├── section_heading
├── figures
│   ├── kind
│   ├── caption
│   ├── description
│   └── stated_values
├── tables
│   ├── caption
│   └── markdown
├── entities
└── visual_only
```

The extraction instructions explicitly require transcription rather than
speculation. Tables are preserved as Markdown and figures include
concrete labels, units, and visible values where available.

### 3. Chunking

Two chunk families are created:

  -----------------------------------------------------------------------
  Modality                Source                  Purpose
  ----------------------- ----------------------- -----------------------
  `text`                  PDF text layer          Precise wording,
                                                  quotations and
                                                  clause-level retrieval

  `visual`                Vision model output     Charts, tables,
                                                  diagrams and
                                                  scanned/visual content
  -----------------------------------------------------------------------

Text chunks use semantic splitting based on adjacent-sentence embedding
distance and a token budget.

Default chunking configuration:

-   Target chunk size: `380` tokens
-   Chunk overlap: `60` tokens
-   Semantic breakpoint percentile: `88`

Tables are kept intact rather than being split across chunks.

### 4. Embeddings

Chunks are embedded using:

``` text
text-embedding-3-small
```

The notebook uses:

``` text
1024 dimensions
```

with L2-normalised vectors so inner product is equivalent to cosine
similarity.

### 5. Hybrid retrieval

The retrieval layer combines:

-   **Dense vector retrieval**
-   **BM25 sparse retrieval**

Dense retrieval is useful for semantic similarity, while BM25 helps with
exact identifiers such as:

-   Product names
-   Statute numbers
-   Tickers
-   SKUs
-   Exact terminology

The two rankings are combined using **Reciprocal Rank Fusion (RRF)**.

Default retrieval configuration:

``` text
Dense top-k:     12
Sparse top-k:    12
RRF k:           60
Final top-k:      6
```

### 6. Vector index

The notebook provides two implementations behind the same interface:

#### NumPy

An exact brute-force inner-product index suitable for smaller corpora.

#### FAISS

A drop-in alternative using `IndexFlatIP` when the corpus becomes
larger.

The same interface is intended to make future migration to a production
vector database straightforward, such as Qdrant, Azure AI Search, or
pgvector.

### 7. Multimodal answer generation

The generation stage receives:

1.  Retrieved textual excerpts
2.  The original page images for the highest-ranked pages

The default configuration allows up to:

``` text
3 pages
```

to be re-attached to the synthesis request.

This is important because the vision-generated index representation can
contain transcription errors. The generation model can compare the
extracted representation with the original page image.

### 8. Structured citations

Answers are returned using a structured schema:

``` text
GroundedAnswer
├── answer
├── citations
│   ├── doc_id
│   ├── page_no
│   └── quote
├── sufficient_context
└── used_visual_evidence
```

The generation instructions require factual claims to be supported by
the retrieved document context.

The notebook also validates citations against the retrieved page set and
logs citations that refer to pages that were not retrieved.

## Evaluation

The project separates evaluation into two independent layers.

### Retrieval evaluation

Retrieval is evaluated using:

-   Recall@K
-   Mean Reciprocal Rank (MRR)

This answers questions such as:

> Did the system retrieve the page containing the required information?

If retrieval recall is poor, improving the generation prompt will not
solve the underlying problem.

### Generation evaluation

Generated answers are evaluated using:

-   Groundedness
-   Correctness
-   Citation precision

An LLM judge scores groundedness and correctness on a `1–5` scale.

Citation precision is checked deterministically by verifying that cited
pages exist in the retrieved set.

The evaluation also tracks whether an example:

-   Requires visual evidence
-   Actually used visual evidence

### Recommended evaluation set

The notebook explicitly recommends adding approximately **30--50
evaluation cases** before trusting aggregate metrics.

LLM judge scores should be treated as a regression signal rather than
ground truth and should be spot-checked against human labels.

## Example

The included notebook runs against a sample document:

``` text
northwind_fy2025_sample.pdf
```

The example document contains six pages, including visual content.

The observed ingestion run produced:

``` text
6 pages
1 text-poor page
6 pages successfully vision-extracted
22 chunks
12 text chunks
10 visual chunks
```

A sample question is:

``` text
What does the main chart show, and what are its axis units?
```

The system retrieves the relevant visual figure and page, then produces
an answer containing:

-   The chart description
-   Axis units
-   Page-level citations
-   A `visual=True` indicator
-   A retrieval trace showing dense and sparse ranks

## Ablation: Text-only vs Multimodal

The notebook includes an ablation experiment that removes visual chunks
and compares the resulting text-only corpus with the full multimodal
corpus.

The current notebook output shows:

  Metric                 Text-only   Multimodal
  -------------------- ----------- ------------
  Recall@6                     1.0          1.0
  MRR                          1.0         0.25
  Citation precision           1.0          1.0
  Groundedness mean            2.0          5.0
  Correctness mean             5.0          5.0

This is a **single evaluation case**, so these numbers should not be
interpreted as production-level performance. The notebook itself
recommends expanding the evaluation set before drawing conclusions.

The main purpose of the ablation is to demonstrate that visual
information can improve groundedness when the answer depends on document
imagery.

## Models

The notebook centralises model configuration in a `Config` dataclass.

The configured models are:

  Component           Model
  ------------------- --------------------------
  Vision extraction   `gpt-5.6-luna`
  Answer synthesis    `gpt-5.6-terra`
  Evaluation judge    `gpt-5.6-sol`
  Embeddings          `text-embedding-3-small`

Model names and pricing should be verified against the current OpenAI
API documentation before running the project in a production
environment.

## Installation

The notebook lists the following dependencies:

``` bash
pip install -q "openai>=1.60" pymupdf pydantic numpy rank-bm25 pillow tiktoken faiss-cpu
```

`faiss` and `tiktoken` are optional in the notebook:

-   Without `faiss`, the system can use the NumPy exact index.
-   Without `tiktoken`, the system falls back to an approximate
    token-counting heuristic.

The notebook was executed with:

``` text
Python 3.11.12
```

## Configuration

Before running the notebook, configure the OpenAI API key.

macOS/Linux:

``` bash
export OPENAI_API_KEY="your-api-key"
```

Do **not** hard-code API keys inside the notebook or commit them to Git.

The main configuration is controlled through the `Config` dataclass:

``` python
@dataclass(frozen=True)
class Config:
    vision_model: str = "gpt-5.6-luna"
    synthesis_model: str = "gpt-5.6-terra"
    judge_model: str = "gpt-5.6-sol"
    embedding_model: str = "text-embedding-3-small"
```

Other configurable parameters include:

-   Image rendering resolution
-   Image maximum size
-   Chunk size
-   Chunk overlap
-   Semantic breakpoint
-   Dense retrieval `top_k`
-   Sparse retrieval `top_k`
-   RRF constant
-   Final retrieval `top_k`
-   Maximum pages supplied to synthesis
-   Maximum concurrent workers
-   Retry count
-   Request timeout

## Running the Notebook

1.  Install the dependencies.
2.  Set `OPENAI_API_KEY`.
3.  Open `multimodal_document_rag.ipynb`.
4.  Update the PDF path:

``` python
PDF_PATH = "/path/to/your/document.pdf"
```

5.  Run the ingestion pipeline:

``` python
pages = load_pdf(PDF_PATH)
extractions = extract_all(pages)
chunks = build_chunks(extractions)
corpus = build_corpus(chunks, pages)
```

6.  Ask a question:

``` python
QUESTION = "What does the main chart show, and what are its axis units?"

answer, hits = answer_question(corpus, QUESTION)
```

7.  Inspect:

``` python
print(answer.answer)
print(answer.sufficient_context)
print(answer.used_visual_evidence)
```

8.  Inspect the retrieval trace to understand which chunks and pages
    were retrieved.

## Cost and Reliability Controls

The notebook includes an instrumented API client with:

-   Explicit retry handling
-   Exponential backoff with jitter
-   Request timeout
-   Token usage tracking
-   Per-model cost accounting
-   Latency reporting

Retryable failures include:

-   Rate-limit errors
-   API timeouts
-   Connection errors
-   Internal server errors

Malformed requests are intentionally not retried.

The usage ledger reports:

``` text
Total calls
Total cost
P50 latency
P95 latency
Input/output tokens
Cost by model
```

This makes ingestion and evaluation cost visible instead of hidden
inside notebook execution.

## Production Considerations

The notebook is designed as a working reference implementation rather
than a complete production service.

For production deployment, consider:

### Persistent storage

Replace in-memory:

-   NumPy vectors
-   BM25 index
-   Page image storage
-   Chunk metadata

with persistent infrastructure.

### Vector search

Consider a production vector database or managed search service when the
corpus becomes too large for brute-force search.

### Object storage

Page images should generally be stored in object storage rather than
embedded indefinitely in application memory.

### Incremental ingestion

Add document hashing/versioning so unchanged documents do not need to be
processed again.

### Observability

Persist:

-   Retrieval traces
-   Model latency
-   Token usage
-   Costs
-   Evaluation metrics
-   Failed page extractions
-   Citation validation failures

### Security

For sensitive documents:

-   Keep API keys outside source code.
-   Apply appropriate document access controls.
-   Avoid logging raw document content.
-   Consider encryption and retention policies.
-   Ensure the selected model/API configuration is appropriate for the
    data classification.

## Design Trade-offs

### Why convert visual content to text for retrieval?

A unified text representation allows the same dense and sparse retrieval
infrastructure to handle both normal text and visual content.

Advantages:

-   Simple retrieval architecture
-   Easy debugging
-   BM25 compatibility
-   Straightforward embedding
-   Easy inspection of retrieved chunks

The trade-off is that the vision model's transcription can introduce
errors.

### Why send original page images during generation?

The original image provides a second source of truth.

The model can verify:

``` text
PDF page → visual content → indexed description → retrieved chunk → original page image → answer
```

This reduces the risk of relying entirely on an intermediate VLM
transcription.

### Why use hybrid retrieval?

Dense retrieval handles semantic similarity, while BM25 is particularly
valuable for exact strings and identifiers.

RRF combines their rankings without requiring score normalisation.

### Why structured output?

Structured Pydantic schemas make it easier to:

-   Validate model output
-   Detect missing fields
-   Validate citations
-   Store results
-   Run deterministic evaluation
-   Prevent free-form citation fabrication

## Limitations

Current limitations include:

-   The implementation is notebook-based.
-   Corpus state is held in memory.
-   The example evaluation set is very small.
-   The vision extraction step adds API cost and latency.
-   Page-level images increase multimodal context size.
-   VLM transcription can still contain errors.
-   Production authentication, persistence, monitoring, and access
    control are not implemented in the notebook.
-   Pricing values in the notebook should be rechecked against current
    API pricing before relying on cost estimates.

## Repository Structure

A minimal repository can be organised as:

``` text
.
├── README.md
├── multimodal_document_rag.ipynb
├── data/
│   └── *.pdf
├── evaluation/
│   └── evaluation_cases.json
└── outputs/
    └── evaluation_results.json
```

The current project is primarily implemented in:

``` text
multimodal_document_rag.ipynb
```

## Key Takeaways

-   **Text-only RAG is not sufficient for visually rich PDFs.**
-   Visual content can be converted into structured textual
    representations for retrieval.
-   The original page image should still be supplied to the multimodal
    generator for verification.
-   Hybrid dense + BM25 retrieval improves robustness for both semantic
    and exact-match queries.
-   RRF avoids fragile score normalisation between retrieval methods.
-   Retrieval and generation should be evaluated separately.
-   Citation validation should be deterministic wherever possible.
-   Multimodal ingestion should be justified through an ablation rather
    than assumed to be beneficial.
-   Cost, latency, retries, and token usage should be measured as
    first-class production metrics.

