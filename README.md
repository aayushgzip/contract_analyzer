# GenAI Legal Document Simplifier

A Generative AI system for **clause-level analysis, simplification, and risk detection** in legal documents such as contracts, agreements, and policies.

---

## 🔧 Core Features

- **Clause Segmentation**
  - Splits legal documents into structured clauses for independent analysis.

- **Plain-Language Explanation**
  - Converts legal jargon into concise, user-friendly explanations.
  - Preserves legal intent while simplifying phrasing.

- **Red-Flag Detection**
  - Identifies potentially risky clauses (penalties, non-competes, hidden charges).
  - Benchmarks clauses against similar contract types.

- **Pre-Signing Checklist Generator**
  - Produces actionable clarification points based on missing or ambiguous terms.

- **Confidence & Fairness Scoring**
  - Scores documents based on clarity, balance, and risk indicators.
  - Suggests safer alternative clause wording (non-advisory).

---

## 🧠 Technical Approach

- **LLM-based Clause Understanding**
- **Retrieval-Augmented Generation (RAG)** for legal grounding
- **Embedding similarity search** for clause benchmarking
- **Prompt-based risk classification**
- **Structured output generation (JSON-based)**

---

## 🛠 Tech Stack

- **Language Models:** Domain-adapted LLMs (Legal / Instruction-tuned)
- **Embeddings:** Sentence / legal-domain embeddings
- **Vector Store:** FAISS / Chroma
- **Backend:** Python
- **NLP:** Transformers, token classification
- **Document Parsing:** PDF / DOCX preprocessing

---

## 📊 Datasets

- Indian Legal Text Fine-Tuning Dataset  
- Indian Supreme Court Judgments  
- Indian Legal Documents Corpus  

(Used for retrieval, benchmarking, and domain adaptation)

---

## 🚀 Future Improvements

- Contract outcome prediction
- Negotiation counter-clause generation
- Cross-document contradiction detection
- Multilingual legal support

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only** and does not provide legal advice.
