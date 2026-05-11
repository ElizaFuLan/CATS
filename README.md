## CATS Evaluation (Quick Start)

This `Supplementary_Material/` folder is for **quickly running CATS evaluation**.

### Install

```bash
pip install -r requirements.txt
```

### Prepare paths

You need:

- **Base model**: `MODEL_PATH` (e.g., a Vicuna/LLaMA-style `AutoModelForCausalLM` checkpoint)
- **Draft adapter**: `DRAFT_ADAPTER_PATH`
- **Shallow adapter**: `SHALLOW_ADAPTER_PATH`

The default question file is `data/question_mtbench.jsonl` (see `evaluation/CATS_dynamic.py`).

### Mode notes (chain vs tree)

- **Chain mode**:
  - `--tree-topk 1`
  - `--total-tokens -1`
- **Tree mode**:
  - `--tree-topk 10`
  - `--total-tokens 40`

### Run evaluation

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$SCRIPT_DIR" python3 evaluation/CATS_dynamic.py \
  --model-path "$MODEL_PATH" \
  --draft-adapter-path "$DRAFT_ADAPTER_PATH" \
  --shallow-adapter-path "$SHALLOW_ADAPTER_PATH" \
  --model-id "cats_chain_${BENCH}_Q0-5" \
  --draft-layer 3 \
  --shallow-layer 15 \
  --threshold 0.0 \
  --steps 5 \
  --sv-passes 1 \
  --tree-topk 1 \
  --total-tokens -1 \
  --num-runs 1 \
  --typical-tau 0.0 --typical-alpha 0.0 --temperature 0.0 \
  --dtype float16 \
  --bench-name "$BENCH" \
  --question-begin 0 --question-end 5 \
  --max-new-tokens 256
```

### Outputs

- **Answers**: `data/<bench-name>/<model-id>/<run>.jsonl`
- **Verification metrics**: `data/<bench-name>/<model-id>/verification_metrics.json`

