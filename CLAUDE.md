# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASearcher is an open-source framework for large-scale online reinforcement learning (RL) training of search agents. It includes:
- Fully asynchronous agentic RL training framework (AReaL)
- Data synthesis agent for generating QA pairs
- Web search agent implementations with tool use capabilities
- Evaluation framework for QA benchmarks (GAIA, xBench-DeepSearch, Frames)

## Common Development Commands

### Running Tests
```bash
# Run all tests with pytest (from AReaL directory)
cd AReaL
python -m pytest -s areal/tests/

# Run tests in Docker environment with GPU support
bash ci/test_areal.sh
```

### Code Formatting
The project uses pre-commit hooks for code formatting:
```bash
# Install pre-commit hooks
pre-commit install

# Run formatting on all files
pre-commit run --all-files

# Individual formatters:
# - Python: black (line-length=88), isort (profile=black), autoflake
# - C++/CUDA: clang-format
# - Markdown: mdformat (wrap=88)
```

### Evaluation
```bash
cd evaluation/

# Set environment variables
export SERPER_API_KEY=your_serper_api_key
export JINA_API_KEY=your_jina_api_key

# Run evaluation script
PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
python3 search_eval_async.py \
    --data_names GAIA,xbench-deepsearch,Frames \
    --model_name_or_path /path/to/model \
    --agent-type asearcher-reasoning \
    --search-client-type async-web-search-access \
    --tensor_parallel_size 4 \
    --temperature 0.6
```

### Training

#### Local Training (Single Node)
```bash
cd AReaL

export SERPER_API_KEY=YOUR_SERPER_API_KEY
export JINA_API_KEY=YOUR_JINA_API_KEY

python3 -m areal.launcher.local ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_web.yaml \
    experiment_name=<experiment_name> \
    trial_name=<trial_name>
```

#### Distributed Training (Multi-Node with Ray)
```bash
cd AReaL

python3 -m areal.launcher.ray ASearcher/train/asearcher.py \
    --config ASearcher/configs/asearcher_web_16nodes.yaml \
    experiment_name=<experiment_name> \
    trial_name=<trial_name> \
    allocation_mode=sglang.d96p1t1+d32p1t1 \
    cluster.n_nodes=16 \
    cluster.n_gpus_per_node=8
```

### Local Retrieval Server (for QA synthesis)
```bash
python3 tools/local_retrieval_server.py \
    --index_path $WIKI2018_WORK_DIR/e5.index/e5_Flat.index \
    --corpus_path $WIKI2018_WORK_DIR/wiki_corpus.jsonl \
    --pages_path $WIKI2018_WORK_DIR/wiki_webpages.jsonl \
    --retriever_name e5 \
    --retriever_model intfloat__e5-base-v2 \
    --faiss_gpu --port 8080
```

## Code Architecture

### Main Components

#### ASearcher Module (`AReaL/ASearcher/`)
- `train/asearcher.py`: Main training workflow for search agents
- `train/search_agent.py`: Search agent implementation with tool use
- `utils/search_tool.py`: Search toolbox for web search and retrieval
- `utils/rewards.py`: Reward calculation for RL training
- `configs/`: YAML configuration files for training

#### AReaL Framework (`AReaL/areal/`)
- `engine/`: Training engines (FSDP, SGLang integration)
- `launcher/`: Distributed training launchers (local, ray, slurm)
- `workflow/`: RL training workflows (multi-turn, vision)
- `utils/`: Utilities for distributed training, data handling

#### Evaluation (`evaluation/`)
- `search_eval_async.py`: Main evaluation script for search agents
- `llm_as_judge.py`: LLM-based evaluation metrics
- Agent types: `asearcher`, `asearcher-reasoning`, `search-r1`
- Search client types: `async-web-search-access`, `async-online-search-access`

#### Agent Implementations (`agent/`)
- `asearcher.py`: Base ASearcher agent
- `asearcher_reasoning.py`: ASearcher with reasoning capabilities
- `search_r1.py`: Search-R1 agent implementation

### Key Design Patterns

1. **Asynchronous RL Training**: Decoupled trajectory collection from model training to handle variable-length trajectories efficiently
2. **Tool Use Architecture**: Agents interact with SearchToolBox for web search (Serper API) and content extraction (Jina API)
3. **Multi-turn Dialogue**: Agents maintain conversation history and can perform up to 128 turns of interaction
4. **Reward Calculation**: F1-based rewards comparing agent answers with ground truth

## Dependencies

- PyTorch > 2.0.0
- Transformers == 4.53.1
- SGLang for inference serving
- Ray for distributed training
- FSDP for model parallelism
- External APIs: Serper (web search), Jina (content extraction)

## Environment Variables

Required for search functionality:
- `SERPER_API_KEY`: API key for Serper web search
- `JINA_API_KEY`: API key for Jina content extraction

Optional:
- `PYTHONPATH`: Should include project root
- `TOKENIZERS_PARALLELISM`: Set to `false` to avoid warnings