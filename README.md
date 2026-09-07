# Teacher-Student LLM Distillation Lab

Notebook-first project for distilling a larger causal language model into a smaller student model. The pipeline uses a teacher model to provide soft next-token supervision, then trains a compact student with a mix of hard language-modeling loss and teacher-distribution matching.

The default setup keeps the run approachable on a single machine:

- teacher: `gpt2-medium`
- student: `distilgpt2`
- dataset: `databricks/databricks-dolly-15k`
- framework: PyTorch + Hugging Face Transformers

Both models share the GPT-2 tokenizer, which makes token-level distillation practical without vocabulary alignment.

## Workflow

The project is intentionally notebook-based. Each notebook is a separate stage of the experiment:

1. prepare a sampled instruction dataset
2. cache top-k teacher logits for response tokens
3. train the student with hard CE + soft teacher loss
4. compare held-out generations
5. run a small inference demo
6. inspect the math behind temperature, loss weighting, and cache size

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
├── generation_metrics.py
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

The core loss pieces are also available in `scripts/distillation_losses.py` for small tests or standalone experiments. The helper implements prompt-token masking, temperature-scaled KL, top-k cache KL, hard CE, and the blended loss used by the student notebook.

The same helper includes a small linear schedule for `α` and `T`. A typical run can begin with more hard-label grounding, then increase the teacher KL weight once the student is stable.

## Outputs

Generated files are kept out of git:

- `artifacts/prepared/` contains sampled JSONL splits.
- `artifacts/teacher_cache/` contains compressed top-k teacher logits and a small cache manifest.
- `checkpoints/student/` contains the distilled student model.
- `reports/` contains training curves, metrics, generation comparisons, cache audits, diagnostics, and compact review CSVs for held-out outputs.

## Cache audit

Before training the student, the teacher cache can be checked without loading either model:

```bash
python scripts/audit_teacher_cache.py --cache-dir artifacts/teacher_cache --output reports/cache_audit.json
```

The audit reports cached row counts, average sequence length, average supervised response tokens, top-k width, teacher-token coverage, estimated cache size, missing teacher logits, top-k index/logit shape mismatches, non-finite logits, loss-mask length mismatches, and empty-supervision warnings. It is a small sanity check that catches common pipeline mistakes before spending time on student training.

## Generation metrics

Held-out teacher/student generations can be summarized without loading either model:

```bash
python scripts/generation_metrics.py \
  --input reports/generation_comparisons.jsonl \
  --output reports/generation_metrics.json
```

The report tracks length ratio, unigram/bigram overlap, teacher/student coverage, novel-token rate, distinct-n, repetition rate, and optional reference overlap when a target response is available.

## Notes

The notebooks default to small runs because teacher forward passes are the expensive part of the workflow. For a larger experiment, increase `train_size` in `00_data_preparation.ipynb`, then increase `max_train_records` in `01_teacher_cache.ipynb`. The student notebook can then run longer by changing epochs, batch size, accumulation steps, temperature, and `distill_alpha`.
