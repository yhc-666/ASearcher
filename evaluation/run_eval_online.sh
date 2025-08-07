set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


MODEL_PATH=/storage/openpsi/experiments/checkpoints/admin/gjx-online-search-qwen2.5-7b-base/20250730-32turns-rF1xformat-1-top5-128x16-gr-mix35k-valid0.3-1/actor/epoch2epochstep82globalstep360

DATA_DIR=/storage/openpsi/users/hechuyi/Agent/agent_eval@master/data

SPLIT=1
MAX_GEN_TOKENS=4098
DATA_NAMES=GAIA,frames
AGENT_TYPE=asearcher
PROMPT_TYPE=asearcher
SEARCH_CLIENT_TYPE=async-web-search-access
temperature=0.6
top_p=0.95
top_k=-1

echo "MODEL PATH: $MODEL_PATH"
echo "Temperature: ${temperature}"
echo "top_p: ${top_p}"
echo "top_k: ${top_k}"
echo "split: ${SPLIT}/"

TOKENIZERS_PARALLELISM=false \
PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
python3 search_eval_async.py \
    --data_names ${DATA_NAMES} \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${MODEL_PATH} \
    --prompt_type $PROMPT_TYPE \
    --agent-type ${AGENT_TYPE} \
    --data_dir ${DATA_DIR} \
    --split test \
    --search-client-type ${SEARCH_CLIENT_TYPE} \
    --max-tokens-per-call ${MAX_GEN_TOKENS} \
    --tensor_parallel_size 4 \
    --num_test_sample -1 \
    --n_sampling 1 \
    --temperature ${temperature} \
    --top_p $top_p \
    --top_k $top_k \
    --start 0 \
    --end -1 \
    --seed 1 \
    --parallel-mode seed \
    --use-jina \
    --llm_as_judge \
    --pass-at-k 2 \
