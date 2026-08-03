# LLM Knowledge Distillation

Notebook-first project for distilling a larger causal language model into a smaller student model. The pipeline uses a teacher model to provide soft next-token supervision, then trains a compact student with a mix of hard language-modeling loss and teacher-distribution matching.

The default setup keeps the run approachable on a single machine:

- teacher: `gpt2-medium`
- student: `distilgpt2`
- dataset: `databricks/databricks-dolly-15k`
- framework: PyTorch + Hugging Face Transformers

Both models share the GPT-2 tokenizer, which makes token-level distillation practical without vocabulary alignment.

## Project structure

```text
notebooks/
├── 00_data_preparation.ipynb
├── 01_teacher_cache.ipynb
├── 02_student_distillation.ipynb
├── 03_evaluation.ipynb
└── 04_inference_demo.ipynb
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name kd-llm
```

Run the notebooks in order. The first pass can use small sample limits to check the full pipeline quickly before increasing the training budget.

## Distillation objective

For each response token, the student minimizes:

```text
L = α · T² · KL(p_teacher^T || p_student^T) + (1 - α) · CE(y, p_student)
```

`T` is the temperature used to soften the teacher distribution, `α` controls how much the student follows the teacher, and the hard cross-entropy term keeps the model grounded in the original response text.

## Outputs

Generated files are kept out of git:

- `artifacts/prepared/` contains sampled JSONL splits.
- `artifacts/teacher_cache/` contains compressed top-k teacher logits.
- `checkpoints/student/` contains the distilled student model.
- `reports/` contains training curves, metrics, and qualitative comparisons.
