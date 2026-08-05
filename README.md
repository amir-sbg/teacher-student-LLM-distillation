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

## Outputs

Generated files are kept out of git:

- `artifacts/prepared/` contains sampled JSONL splits.
- `artifacts/teacher_cache/` contains compressed top-k teacher logits and a small cache manifest.
- `checkpoints/student/` contains the distilled student model.
- `reports/` contains training curves, metrics, generation comparisons, and a compact review CSV for held-out outputs.

## Notes

The notebooks default to small runs because teacher forward passes are the expensive part of the workflow. For a larger experiment, increase `train_size` in `00_data_preparation.ipynb`, then increase `max_train_records` in `01_teacher_cache.ipynb`. The student notebook can then run longer by changing epochs, batch size, accumulation steps, temperature, and `distill_alpha`.
