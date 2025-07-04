import gradio as gr
from qa_core import answer_question

def qa_interface(user_input):
    if not user_input.strip():
        return "❗ 질문을 입력해 주세요."
    return answer_question(user_input)

gr.Interface(
    fn=qa_interface,
    inputs=gr.Textbox(lines=2, placeholder="질문을 입력하세요...", label="질문"),
    outputs=gr.Textbox(label="🤖 답변"),
    title="🩺 한글 의료 지식 기반 LLM 질의응답 시스템",
    description="FAISS + Qwen 모델 기반 QA 데모입니다.",
).launch(share=True)
