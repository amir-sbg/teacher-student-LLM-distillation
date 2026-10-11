# Teacher-Student LLM Distillation

Notebook-first project for distilling a larger causal language model into a smaller student model. The experiment stays small enough to run locally, but it still covers the important parts of practical LLM distillation: data preparation, teacher-logit caching, soft-label training, generation review, calibration checks, and run summaries.

The default setup keeps the run approachable on a single machine:

- teacher: `gpt2-medium`
- student: `distilgpt2`
- dataset: `databricks/databricks-dolly-15k`
- framework: PyTorch + Hugging Face Transformers

Both models share the GPT-2 tokenizer, so token-level KL can be computed directly without a vocabulary-alignment layer.

## Workflow

The project is intentionally notebook-based. Each notebook is a stage of the experiment:

1. prepare a sampled instruction dataset
2. cache top-k teacher logits for response tokens
3. train the student with hard CE + soft teacher loss
4. compare held-out generations
5. run a small inference demo
6. inspect temperature, loss weighting, calibration, and cache-size tradeoffs

## Project structure

```text
notebooks/
├── 00_data_preparation.ipynb
├── 01_teacher_cache.ipynb
├── 02_student_distillation.ipynb
├── 03_evaluation.ipynb
├── 04_inference_demo.ipynb
└── 05_distillation_math.ipynb
scripts/
├── audit_teacher_cache.py
├── calibration_metrics.py
├── generation_metrics.py
├── training_summary.py
└── distillation_losses.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name kd-llm
```

Run the notebooks in order. The first pass can use the default small sample limits to check the full pipeline quickly before increasing the training budget.

## Distillation objective

For each response token, the student minimizes:

```text
L = α · T² · KL(p_teacher^T || p_student^T) + (1 - α) · CE(y, p_student)
```

`T` is the temperature used to soften the teacher distribution, `α` controls how much the student follows the teacher, and the hard cross-entropy term keeps the model grounded in the original response text. The teacher cache stores only top-k logits per response position, which preserves useful dark knowledge while keeping storage manageable.

The core loss pieces are also available in `scripts/distillation_losses.py` for small tests or standalone experiments. The helper implements prompt-token masking, temperature-scaled KL, confidence-weighted teacher matching, Jensen-Shannon soft-target matching, top-k cache KL against the student's full-vocabulary distribution, hard CE, entropy diagnostics, and the blended loss used by the student notebook. Teacher-token rank diagnostics add top-k agreement, mean reciprocal rank, and mean rank so student imitation can be inspected beyond loss alone.

The same helper includes a small linear schedule for `α` and `T`. A typical run can begin with more hard-label grounding, then increase the teacher KL weight once the student is stable. Linear CKA is included for experiments that align intermediate student and teacher representations even when their hidden widths differ.

## Outputs

Generated files are kept out of git:

- `artifacts/prepared/` contains sampled JSONL splits.
- `artifacts/teacher_cache/` contains compressed top-k teacher logits and a small cache manifest.
- `checkpoints/student/` contains the distilled student model.
- `reports/` contains training curves, metrics, generation comparisons, cache audits, calibration diagnostics, and compact review CSVs for held-out outputs.

## Cache audit

Before training the student, the teacher cache can be checked without loading either model:

```bash
python scripts/audit_teacher_cache.py --cache-dir artifacts/teacher_cache --output reports/cache_audit.json
```

The audit reports cached row counts, average sequence length, average supervised response tokens, top-k width, teacher-token coverage, estimated cache size, missing teacher logits, top-k index/logit shape mismatches, non-finite logits, loss-mask length mismatches, and empty-supervision warnings. It also returns a `pass` or `warning` status; add `--fail-on-warning` when using the audit in a local CI check before spending time on student training.

## Generation metrics

Held-out teacher/student generations can be summarized without loading either model:

```bash
python scripts/generation_metrics.py \
  --input reports/generation_comparisons.jsonl \
  --output reports/generation_metrics.json
```

The report tracks length ratio, unigram/bigram overlap, Jaccard overlap, teacher/student coverage, novel-token rate, distinct-n, repetition rate, and optional reference overlap when a target response is available. It also adds lightweight quality flags for low teacher overlap, over-copying, very short or long answers, and repetition-heavy student generations.

## Run summary and calibration

After training, a metric history can be reduced into a small summary:

```bash
python scripts/training_summary.py \
  --input reports/training_history.jsonl \
  --output reports/training_summary.json
```

If confidence/correctness rows are available, calibration can be checked with:

```bash
python scripts/calibration_metrics.py \
  --input reports/confidence_rows.jsonl \
  --output reports/calibration_metrics.json
```

## Notes

The notebooks default to small runs because teacher forward passes are the expensive part of the workflow. For a larger experiment, increase `train_size` in `00_data_preparation.ipynb`, then increase `max_train_records` in `01_teacher_cache.ipynb`. The student notebook can then run longer by changing epochs, batch size, accumulation steps, temperature, and `distill_alpha`.
