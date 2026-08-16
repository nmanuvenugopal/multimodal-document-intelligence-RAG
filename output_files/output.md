# Multimodal Document RAG --- Output & Observations

This document summarizes the observed output from the multimodal
document RAG pipeline, including document extraction, retrieval, answer
generation, evaluation, cost, and latency.

## 1. Document Ingestion and Extraction

``` text
2026-08-15 22:30:52,355 | INFO | mm_rag | Loaded northwind_fy2025_sample: 6 pages (1 text-poor)
2026-08-15 22:30:57,473 | INFO | mm_rag | Extracted 6/6 pages
2026-08-15 22:31:00,230 | INFO | mm_rag | Built 22 chunks (12 text, 10 visual)
```

  Metric                           Result
  ------------------------------ --------
  Total pages                           6
  Text-poor pages                       1
  Successfully extracted pages        6/6
  Total chunks                         22
  Text chunks                          12
  Visual chunks                        10

### Observation

All six pages were successfully processed. One page was identified as
text-poor, showing the value of the visual extraction path for pages
where useful information may not be available through the PDF text
layer.

The resulting corpus contains both textual and visual representations:
**12 text chunks** and **10 visual chunks**.

## 2. Ingestion Cost and Latency

``` json
{
  "calls": 12,
  "cost_usd": 0.007,
  "latency_p50_s": 3.362,
  "latency_p95_s": 5.116,
  "by_model": {
    "gpt-5.6-luna": {
      "calls": 6,
      "in": 15601,
      "out": 1533,
      "usd": 0.007
    },
    "text-embedding-3-small": {
      "calls": 6,
      "in": 2874,
      "out": 0,
      "usd": 0.0001
    }
  }
}
```

  Model                        Calls   Input Tokens   Output Tokens       Cost
  -------------------------- ------- -------------- --------------- ----------
  `gpt-5.6-luna`                   6         15,601           1,533    \$0.007
  `text-embedding-3-small`         6          2,874               0   \$0.0001

The reported ingestion cost is approximately **\$0.007**, with P50
latency of **3.362 seconds** and P95 latency of **5.116 seconds**.

## 3. Generated Answer

The pipeline was tested with a question asking what the main chart shows
and what its axis units are.

The generated answer identifies **Figure 2** as a bar chart of FY2025
gross margin by product line:

  Product Line              Gross Margin
  ----------------------- --------------
  Retail Analytics                 41.2%
  Wholesale Data                   28.7%
  Digital Subscriptions            63.5%
  Professional Services            22.4%
  Group average                    38.9%

The y-axis is identified as **Gross margin (%)**.

``` text
sufficient_context: True | visual: True
```

### Observation

The system determined that the retrieved context was sufficient and
explicitly reported that **visual evidence was used** to answer the
question.

## 4. Citations

``` text
northwind_fy2025_sample p.4: Figure 2: Gross margin by product line, FY2025
northwind_fy2025_sample p.4: Gross margin (%)
northwind_fy2025_sample p.4: Group avg 38.9%
```

All displayed citations point to **page 4**, where the relevant chart is
located. They support the chart identity, metric/unit, and group-average
value.

## 5. Retrieval Trace

``` text
0.0328 | visual figure  | p.4 | dense=1  sparse=1 | CHART — Figure 2: Gross margin by product line, FY2025
0.0320 | visual summary | p.4 | dense=2  sparse=3 | This page presents gross margin analysis by product line for FY2025
0.0313 | visual figure  | p.2 | dense=6  sparse=2 | CHART — Figure 1: Monthly active users, FY2025
0.0301 | text   prose   | p.1 | dense=9  sparse=4 | Basis of preparation...
0.0294 | text   prose   | p.3 | dense=10 sparse=6 | Northwind Analytics Ltd | Annual Review FY2025...
0.0294 | visual figure  | p.6 | dense=8  sparse=8 | DIAGRAM — Figure 3: Target data platform architecture...
```

### Retrieval Observation

The most relevant result was the **visual figure from page 4**:

``` text
dense rank = 1
sparse rank = 1
```

Both dense and sparse retrieval independently ranked the correct visual
chunk first.

The second result was also from page 4 and represented the visual page
summary. This shows that the hybrid retrieval system surfaced both the
detailed figure representation and supporting page-level context.

## 6. Text-Only vs Multimodal Evaluation

  Metric                 Text-only   Multimodal
  -------------------- ----------- ------------
  Recall@6                     1.0          1.0
  MRR                          1.0         0.25
  Citation Precision           1.0          1.0
  Groundedness Mean            2.0          5.0
  Correctness Mean             5.0          5.0

### Recall@6

Both approaches achieved:

``` text
Recall@6 = 1.0
```

A relevant page therefore appeared within the top six retrieved results
in the displayed evaluation.

### Mean Reciprocal Rank

``` text
Text-only:  1.0
Multimodal: 0.25
```

For the displayed evaluation case, the text-only system has the higher
MRR.

### Citation Precision

``` text
Text-only:  1.0
Multimodal: 1.0
```

Both approaches achieved perfect citation precision in the displayed
evaluation.

### Groundedness

``` text
Text-only:  2.0
Multimodal: 5.0
```

This is the largest improvement shown in the results. For this visual
question, the multimodal pipeline produced an answer that the evaluator
considered substantially better grounded in the supplied document
evidence.

### Correctness

``` text
Text-only:  5.0
Multimodal: 5.0
```

Both approaches received the maximum displayed correctness score.

## 7. Evaluation Cost and Latency

The cumulative output after evaluation reports:

``` text
calls:           21
cost_usd:        0.0576
latency_p50_s:   1.946
latency_p95_s:   5.106
```

Visible model usage includes:

``` text
gpt-5.6-luna
  calls: 6
  input tokens: 15,601
  output tokens: 1,533
  cost: $0.007

text-embedding-3-small
  calls: 10
  input tokens: 4,147
  output tokens: 0
  cost: $0.0001
```

The overall cumulative cost shown in the evaluation output is
**\$0.0576**.

## 8. Key Observations

1.  **Successful document processing** --- all 6 pages were extracted
    successfully.
2.  **Both modalities were indexed** --- 12 text chunks and 10 visual
    chunks were created.
3.  **Visual retrieval worked correctly** --- the relevant page-4 chart
    ranked first in both dense and sparse retrieval.
4.  **Visual evidence was actually used** --- the generated output
    reports `visual: True`.
5.  **Page-level citations were generated** --- the relevant evidence
    was cited from page 4.
6.  **Groundedness improved in the displayed multimodal evaluation** ---
    from **2.0 to 5.0**.
7.  **Correctness remained 5.0** for both text-only and multimodal runs.
8.  **Citation precision remained 1.0** for both approaches.
9.  **Multimodal processing adds cost** because visual extraction,
    answer synthesis, embeddings, and evaluation require additional
    model calls.

## 9. Evaluation Limitation

The displayed comparison should be treated as an **initial pipeline
validation rather than a benchmark**.

The screenshots show only a very small evaluation setup. Therefore,
results such as:

``` text
MRR:          1.0 → 0.25
Groundedness: 2.0 → 5.0
```

are useful observations for the current test case, but they are not
sufficient to conclude that either approach universally performs better.

A larger evaluation dataset containing both text-based and visual
questions is required for a meaningful comparison.

## 10. Overall Result

The experiment demonstrates that the multimodal RAG pipeline is
functioning end-to-end:

``` text
PDF Document
     ↓
Text + Page Image Extraction
     ↓
Vision-Based Page Understanding
     ↓
Text + Visual Chunking
     ↓
Dense + BM25 Retrieval
     ↓
Reciprocal Rank Fusion
     ↓
Relevant Page / Figure Retrieval
     ↓
Multimodal Answer Generation
     ↓
Grounded Answer + Page Citations
     ↓
Evaluation
```

For the tested chart question, the system successfully retrieved the
correct visual figure, used visual evidence, extracted the requested
numerical information, and returned citations pointing to the relevant
source page.

The most notable result in the displayed evaluation is the increase in
**groundedness from 2.0 for text-only to 5.0 for multimodal**, while the
displayed correctness and citation precision remain at their maximum
values.
