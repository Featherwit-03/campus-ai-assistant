# -*- coding: utf-8 -*-
"""
问答主程序：检索 + 大模型生成答案，提供 Gradio 网页界面。
运行：python app.py
"""
import json
import os
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 6  # 检索时取最相似的几个片段（太大易混入无关内容，让模型误判"没有答案"）

ASSETS_DIR = Path(__file__).parent / "assets"
BANNER = ASSETS_DIR / "banner.jpg"  # 若 assets/ 下有 banner.jpg，会自动作为顶部横幅显示

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 全局只加载一次，避免每次提问都重新加载
_embedder = None
_index = None
_chunks = None
_client = None


def load_resources():
    global _embedder, _index, _chunks, _client
    if _index is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
        # 读取时同样绕过 faiss 对中文路径的支持问题。
        # 新版 faiss 的 deserialize_index 需要 numpy 数组（带 shape），不能直接传字节。
        raw = (INDEX_DIR / "faiss.index").read_bytes()
        _index = faiss.deserialize_index(np.frombuffer(raw, dtype=np.uint8))
        data = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        _chunks = data["chunks"]
    if _client is None:
        assert API_KEY, "请在 .env 中配置 DEEPSEEK_API_KEY（参考 .env.example）"
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _embedder, _index, _chunks, _client


def retrieve(query: str, top_k: int = TOP_K):
    """把问题向量化，在索引里找最相似的片段，返回 [(分数, 片段, 编号), ...]"""
    embedder, index, chunks, _ = load_resources()
    vec = embedder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.asarray(vec, dtype="float32"), top_k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:  # 索引不足 top_k 时的空位
            continue
        results.append((float(score), chunks[i], int(i)))
    return results


def build_prompt(query: str, results):
    """把检索到的资料拼进提示词，让模型只根据资料回答"""
    context = "\n\n".join(
        f"[资料{i + 1}] {chunk}" for i, (_, chunk, _) in enumerate(results)
    )
    return (
        "你是一个校园信息助手。请依据下面提供的资料回答用户的问题，不要编造。\n"
        "只要资料中包含与问题相关的信息，就必须直接回答，不得回复'没有相关信息'或输出【无答案】。\n"
        "如果资料与问题相关但不完整，请基于现有资料尽力回答，并补充说明哪些细节资料中没有。\n"
        "只有当资料与问题完全无关、确实没有任何相关信息时，才在第一行输出【无答案】。\n"
        "请用中文回答，先给出最直接的结论，再分点补充细节；\n"
        "尽量覆盖资料中与该问题相关的所有要点，不要遗漏关键规则（如名额、成绩、时间、条件等）。\n"
        "回答末尾请列出参考的资料编号。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{query}"
    )


def answer(question: str, top_k: int = TOP_K):
    """完整问答流程：检索 → 拼提示词 → 调大模型 → 返回答案和来源"""
    if not question.strip():
        return "请输入问题。"

    embedder, index, chunks, client = load_resources()
    results = retrieve(question, top_k)
    if not results:
        return "知识库是空的，请先运行 python build_index.py 建库。"

    prompt = build_prompt(question, results)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个乐于助人的校园信息助手，请依据给定资料回答用户的问题。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,  # 低温度 = 更稳定、更少自由发挥
    )
    answer_text = resp.choices[0].message.content

    # 资料里没有答案时：不显示参考来源，改为引导向学校公众号的老师咨询
    if "【无答案】" in answer_text:
        return (
            "资料中暂时没有这个问题的相关信息。\n"
            "建议关注学校官方微信公众号，向公众号里的老师咨询，获取准确答复。"
        )

    source_lines = "\n".join(
        f"参考{i + 1}: {chunk[:60]}..." for i, (_, chunk, _) in enumerate(results)
    )
    return answer_text + "\n\n---\n参考来源：\n" + source_lines


def main():
    print("正在加载模型和索引，首次运行会下载模型，请稍等...")
    load_resources()

    # 界面样式：DeepSeek 式排版（顶部横幅 + 大聊天区 + 底部输入），按钮悬停变色
    css = """
    body { background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%); }
    .gradio-container { max-width: 1080px !important; margin: 0 auto; }

    /* 顶部横幅：纯 CSS 渐变，效果等同图片 */
    .hero {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 55%, #7c3aed 100%);
        border-radius: 18px;
        padding: 30px 20px 26px 20px;
        text-align: center;
        color: #ffffff;
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.28);
    }
    .hero::before, .hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
    }
    .hero::before { width: 190px; height: 190px; top: -70px; left: -50px; }
    .hero::after { width: 250px; height: 250px; bottom: -120px; right: -70px; }
    .hero-emoji { font-size: 36px; margin-bottom: 6px; }
    .hero-title { font-size: 27px; font-weight: 700; letter-spacing: 2px; }
    .hero-sub { font-size: 14px; opacity: 0.92; margin-top: 8px; }

    /* 顶部图片横幅（如果 assets/banner.jpg 存在） */
    #banner img {
        width: 100%;
        height: 220px;
        object-fit: cover;
        border-radius: 18px;
        margin-bottom: 20px;
    }

    /* 聊天区：放大 + 气泡样式（Gradio 6 类名：user-message / bot-message） */
    #chatbot { font-size: 16px !important; }
    #chatbot .message-item {
        border-radius: 16px !important;
        padding: 14px 16px !important;
        line-height: 1.6 !important;
    }
    #chatbot .user-message {
        background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
        border-color: #4f46e5 !important;
        color: #ffffff !important;
    }
    #chatbot .user-message .message-text { color: #ffffff !important; }
    #chatbot .bot-message {
        background: #eef2ff !important;
        border-color: #c7d2fe !important;
        color: #1e293b !important;
    }
    #chatbot .bot-message .message-text { color: #1e293b !important; }

    /* 输入框：更大、聚焦光晕 */
    #question-box textarea {
        font-size: 16px !important;
        padding: 12px 14px !important;
        border-radius: 14px !important;
        border: 1px solid #cbd5e1 !important;
        transition: border-color 0.3s, box-shadow 0.3s;
    }
    #question-box textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
    }

    /* 按钮：悬停变色 + 上浮 + 阴影 */
    #send-btn {
        background: #2563eb !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        min-height: 44px !important;
        transition: all 0.3s ease !important;
    }
    #send-btn:hover {
        background: #1d4ed8 !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
    }
    #clear-btn {
        background: #f1f5f9 !important;
        color: #334155 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        min-height: 44px !important;
        transition: all 0.3s ease !important;
    }
    #clear-btn:hover {
        background: #e2e8f0 !important;
        transform: translateY(-2px);
    }
    """

    def chat(history, msg):
        """聊天流程：记录用户问题 → 生成回答 → 追加到对话"""
        history = list(history) if history else []
        msg = (msg or "").strip()
        if not msg:
            return history, ""
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": answer(msg)})
        return history, ""

    with gr.Blocks(title="校园AI问答助手", css=css, theme=gr.themes.Soft()) as demo:
        # 顶部横幅：有图片用图片，没有图片用渐变横幅
        if BANNER.exists():
            gr.Image(value=str(BANNER), show_label=False, container=False, elem_id="banner")
        gr.HTML("""
        <div class="hero">
            <div class="hero-emoji">🎓 🤖</div>
            <div class="hero-title">校园AI问答助手</div>
            <div class="hero-sub">基于学校公开资料回答问题 · 答案带参考来源 · 不编造</div>
        </div>
        """)

        # Gradio 6 的 Chatbot 默认就是消息格式，不再需要 type 参数
        chatbot = gr.Chatbot(height=560, show_label=False, elem_id="chatbot")

        with gr.Row():
            question = gr.Textbox(
                placeholder="输入你的问题，回车或点击发送…",
                lines=2,
                elem_id="question-box",
                scale=8,
                container=False,
            )
            send_btn = gr.Button("发送", variant="primary", size="lg", elem_id="send-btn", scale=1)
            clear_btn = gr.Button("清空", size="lg", elem_id="clear-btn", scale=1)

        gr.Examples(
            examples=[
                "转专业有什么条件？",
                "国家奖学金多少钱？",
                "校园卡丢了怎么办？",
                "升华学生公寓是几人间？",
            ],
            inputs=question,
            label="试试这些问题",
        )

        send_btn.click(fn=chat, inputs=[chatbot, question], outputs=[chatbot, question])
        question.submit(fn=chat, inputs=[chatbot, question], outputs=[chatbot, question])
        clear_btn.click(fn=lambda: ([], ""), inputs=None, outputs=[chatbot, question])

    demo.launch()


if __name__ == "__main__":
    main()
