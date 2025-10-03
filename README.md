# Domain-Specific Fine-Tuning + Deployment 🚀

An end-to-end **AI Engineer portfolio project** demonstrating how to fine-tune, evaluate, optimize, and deploy a domain-specific Large Language Model.

---

## 🔎 Overview  
This project showcases a **full production pipeline** for custom LLMs:  

1. **Fine-Tune (LoRA/PEFT)** – Domain data + lightweight parameter-efficient tuning  
2. **Evaluate** – BLEU, ROUGE, and F1 for quantitative benchmarking  
3. **Optimize** – Quantization for efficient inference  
4. **Export** – Convert to ONNX for cross-platform deployment  
5. **Serve** – FastAPI backend (with ONNX Runtime)  
6. **Deploy** – Dockerized → Google Cloud Run / Render  
7. **Demo** – Next.js frontend (chat-style UI for domain queries)

---

## 🛠️ Tech Stack  
- **Model Training**: HuggingFace Transformers, PEFT (LoRA), PyTorch  
- **Evaluation**: BLEU / ROUGE / F1 scoring scripts  
- **Optimization**: Optimum + ONNX Runtime (quantization + export)  
- **Serving**: FastAPI microservice (REST + CORS for frontend)  
- **Deployment**: Docker → Cloud Run / Render  
- **Frontend**: Next.js + Tailwind (finance-themed chat UI)

---

## 💡 Why It Matters  
- Demonstrates **end-to-end AI engineering** beyond API calls.  
- Covers **both training and deployment**, showing ability to:  
  - Adapt a foundation model with domain knowledge  
  - Evaluate rigorously with metrics  
  - Optimize for real-world inference (quantization, ONNX)  
  - Deploy as a scalable service with a polished frontend  

This project highlights **enterprise readiness** and **full-stack ML skills** recruiters look for.

---

## 🚀 Demo  
- **Frontend (Next.js)**: Finance-oriented chat UI  
- **Backend (FastAPI)**: `/infer` endpoint with ONNX Runtime  

*(Add your deployed demo link here when ready, e.g. Vercel + Render URL)*

---

## 📊 Example Workflow
```bash
# 1) Fine-tune with LoRA
python training/train_lora.py --model qwen-3b --data domain.json

# 2) Evaluate
python evaluation/eval_metrics.py --preds preds.txt --refs refs.txt

# 3) Export + Quantize
python -m optimum.exporters.onnx --model ./finetuned --optimize O3 onnx/

# 4) Serve locally
uvicorn app:app --reload

# 5) Deploy with Docker
docker build -t domain-llm .
docker run -p 8000:8000 domain-llm
```

## 📈 Resume Highlights

Built end-to-end LLM pipeline: fine-tuned domain model, evaluated with BLEU/ROUGE/F1.

Optimized with quantization + ONNX export, achieving faster inference.

Deployed FastAPI backend + Next.js frontend to Cloud Run/Render.

Demonstrates full-stack AI engineering: training → optimization → deployment.

## 📂 Repo Structure
```bash
├── training/            # Fine-tuning scripts (LoRA/PEFT)
├── evaluation/          # Metric scripts (BLEU/ROUGE/F1)
├── serving/             # FastAPI app + ONNX Runtime
├── frontend/            # Next.js demo UI
├── Dockerfile           # Container for backend
├── requirements.txt     
└── README.md
```
## 🔮 Next Steps

Add streaming token output for ChatGPT-style UX

Expand dataset for more robust domain generalization

Deploy with CI/CD (GitHub Actions + Cloud Run)