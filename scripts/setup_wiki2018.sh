#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
WIKI2018_WORK_DIR="${1:-$HOME/wiki2018}"     # 允许通过第1个参数自定义工作目录
AUTO_PATCH_LAUNCH_SH="${AUTO_PATCH_LAUNCH_SH:-1}"  # 是否自动改写 scripts/launch_local_server.sh 中的路径(1=是, 0=否)
LAUNCH_SH_PATH="${LAUNCH_SH_PATH:-scripts/launch_local_server.sh}"  # 你的 launch_local_server.sh 路径

# ===== Prep =====
echo "[*] Work dir: ${WIKI2018_WORK_DIR}"
mkdir -p "${WIKI2018_WORK_DIR}/e5.index"
mkdir -p "${WIKI2018_WORK_DIR}/e5-base-v2"

# 建议开启更快下载（可选）
export HF_HUB_ENABLE_HF_TRANSFER=1

# 安装依赖（如果你已有就会跳过）
python3 -m pip install -U --quiet huggingface_hub datasets

# ===== Download e5-base-v2 model (到本地目录) =====
echo "[*] Downloading embedding model: intfloat/e5-base-v2"
python3 - <<'PY'
from huggingface_hub import snapshot_download
import os
out_dir = os.environ.get("WIKI2018_WORK_DIR")
snapshot_download(
    repo_id="intfloat/e5-base-v2",
    repo_type="model",
    local_dir=os.path.join(out_dir, "e5-base-v2"),
    local_dir_use_symlinks=False
)
print("[+] e5-base-v2 downloaded.")
PY

# ===== Download ASearcher-Local-Knowledge assets =====
echo "[*] Downloading retriever index / corpus / webpages from inclusionAI/ASearcher-Local-Knowledge"
python3 - <<'PY'
from huggingface_hub import hf_hub_download
import os, shutil

base_dir = os.environ.get("WIKI2018_WORK_DIR")
idx_dir  = os.path.join(base_dir, "e5.index")
# 需要的三个文件名
files = [
    ("e5.index", "e5_Flat.index",   os.path.join(idx_dir, "e5_Flat.index")),
    (".",        "wiki_corpus.jsonl",    os.path.join(base_dir, "wiki_corpus.jsonl")),
    (".",        "wiki_webpages.jsonl",  os.path.join(base_dir, "wiki_webpages.jsonl")),
]
for sub, fname, dest in files:
    path = hf_hub_download(
        repo_id="inclusionAI/ASearcher-Local-Knowledge",
        repo_type="dataset",
        filename=(fname if sub=="." else f"{sub}/{fname}")
    )
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(path, dest)
    print(f"[+] saved: {dest}")
PY

# ===== Tree (可选) =====
echo
echo "[*] Final layout:"
echo "${WIKI2018_WORK_DIR}"
find "${WIKI2018_WORK_DIR}" -maxdepth 2 -type f | sed "s#${WIKI2018_WORK_DIR}#  #"


echo
echo "[✓] All done."
echo "Next:"
echo "  1) export RAG_SERVER_ADDR_DIR=/path/to/rag_addr_dir"
echo "  2) bash ${LAUNCH_SH_PATH} 8888 \$RAG_SERVER_ADDR_DIR"
