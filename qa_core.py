import json
import numpy as np
import faiss
from text2vec import SentenceModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载原文和向量索引
texts = json.load(open("text_map.json", encoding="utf-8"))
index = faiss.read_index("faiss_index.bin")

# 加载向量模型
model_embed = SentenceModel("shibing624/text2vec-base-multilingual")

# 加载语言模型（Qwen）
model_id = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 检索函数
def retrieve_context(query, top_k=3):
    vec = model_embed.encode([query])[0]
    D, I = index.search(np.array([vec]).astype("float32"), k=top_k)
    return "\n".join([texts[i] for i in I[0]])

# ✅ 定义答题函数（必须和 app.py 导入名称一致）
def answer_question(query):
    context = retrieve_context(query)
    if not context.strip():
        return "⚠️ 관련된 정보를 찾을 수 없습니다."

    prompt = f"""당신은 전문 의료 지식이 풍부한 AI 어시스턴트입니다.
다음 정보를 참고하여 질문에 대해 구체적이고 정확하게 답변해 주세요.

정보:
{context}

질문: {query}
답변:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "답변:" in result:
        return "🤖 답변: " + result.split("답변:")[-1].strip()
    return "🤖 답변: " + result.strip()
