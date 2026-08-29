# -*- coding: utf-8 -*-
"""
建库脚本：把 data/ 下的资料切分、向量化，存入 FAISS 索引。
运行：python build_index.py
"""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "index"

CHUNK_SIZE = 300      # 每个片段大约多少字
CHUNK_OVERLAP = 50    # 相邻片段重叠多少字，避免切断关键句
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文效果好、体积小的免费本地模型


def load_texts(data_dir: Path):
    """读取 data/ 下所有 .md/.txt 文件，返回 [(文件名, 正文), ...]"""
    texts = []
    for path in sorted(data_dir.glob("*.md")) + sorted(data_dir.glob("*.txt")):
        texts.append((path.name, path.read_text(encoding="utf-8")))
    return texts


def chunk_text(text: str, size: int, overlap: int):
    """按固定长度切分，带重叠，保证语义尽量完整"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def main():
    print("[1/4] 读取资料...")
    files = load_texts(DATA_DIR)
    assert files, f"data 目录下没有资料，请把整理好的 .md/.txt 放进来：{DATA_DIR}"

    chunks, sources = [], []
    for filename, text in files:
        for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)
                sources.append(filename)

    print(f"[2/4] 共切分 {len(chunks)} 个片段，加载嵌入模型（首次运行会自动下载）...")
    model = SentenceTransformer(EMBED_MODEL)

    print("[3/4] 向量化中...")
    vectors = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype="float32")

    print("[4/4] 保存索引...")
    INDEX_DIR.mkdir(exist_ok=True)
    # 内积 + 归一化向量 = 余弦相似度
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    # Windows 下 faiss 直接写中文路径会失败，先序列化，再用 Python 写文件
    index_data = faiss.serialize_index(index)
    (INDEX_DIR / "faiss.index").write_bytes(index_data)

    (INDEX_DIR / "chunks.json").write_text(
        json.dumps({"chunks": chunks, "sources": sources}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成！索引保存在 {INDEX_DIR}，共 {index.ntotal} 个片段。")


if __name__ == "__main__":
    main()
