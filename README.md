# 🩺 Korean Medical QA System (의료 질의응답 시스템)

이 프로젝트는 **FAISS + Qwen 모델**을 기반으로 한 **한국어 의료 질의응답 시스템**입니다. Google Colab에서 실행 가능합니다.

## 🔧 프로젝트 구조
- `app.py` : Gradio 기반 웹 인터페이스 실행 파일
- `qa_core.py` : 벡터 검색 + 응답 생성 로직
- `text_map.json` : 원본 텍스트 단락 매핑 파일 (검색 결과 출력용)
- `faiss_index.bin` : FAISS 벡터 인덱스 파일
- `requirements.txt` : 필요 라이브러리 목록 (예: transformers, text2vec)
- `demo.png` : 시스템 시연 스크린샷 (선택 사항)

## 📦 권장 실행 환경
Google Colab에서 실행하는 것을 권장합니다:

```bash
# 프로젝트 클론
!git clone https://github.com/your-id/korean-medical-qa.git
%cd korean-medical-qa

# 필요한 라이브러리 설치
!pip install -r requirements.txt

# 서비스 실행 (Colab에서 Gradio UI 제공)
!python app.py
