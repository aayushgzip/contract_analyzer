import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
from io import BytesIO
import re
from openai import OpenAI
import os
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="🏛️ Indian Contract Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .risk-high {
        background-color: #000000;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .risk-medium {
        background-color: #000000;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .risk-low {
        background-color: #000000;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .confidence-score {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .chatbot-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)


class ContractAnalyzer:
    def __init__(self):
        self.setup_models()
        self.load_benchmarks()

    def setup_models(self):
        """Initialize InLegalBERT and other models"""
        try:
            # Load InLegalBERT for legal text analysis
            st.info("Loading InLegalBERT model...")
            from transformers import (
                AutoModel,
            )  # Import AutoModel instead of AutoModelForPreTraining

            self.tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
            self.legal_model = AutoModel.from_pretrained(
                "law-ai/InLegalBERT"
            )  # Changed here
            self.legal_pipeline = pipeline("fill-mask", model="law-ai/InLegalBERT")
            st.success("✅ InLegalBERT loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading InLegalBERT: {str(e)}")
            self.tokenizer = None
            self.legal_model = None
            self.legal_pipeline = None

    def load_benchmarks(self):
        """Load benchmark data for contract analysis"""
        self.benchmarks = {
            "employment": {
                "notice_period": {
                    "normal": "30 days",
                    "range": "15-90 days",
                    "unit": "days",
                },
                "probation": {
                    "normal": "3-6 months",
                    "max_acceptable": "12 months",
                    "unit": "months",
                },
                "non_compete": {
                    "normal": "6-12 months",
                    "max_acceptable": "24 months",
                    "unit": "months",
                },
                "bond_penalty": {
                    "normal": "0-2 months salary",
                    "suspicious": ">6 months salary",
                    "unit": "months",
                },
                "severance": {
                    "typical": "0-3 months salary",
                    "suspicious": ">6 months salary",
                },
                "garden_leave": {
                    "typical": "0-3 months",
                    "suspicious": ">6 months",
                    "unit": "months",
                },
            },
            "rental": {
                "security_deposit": {
                    "normal": "2-3 months",
                    "suspicious": ">6 months",
                    "unit": "months",
                },
                "rent_increase": {
                    "normal": "5-10% annually",
                    "suspicious": ">15% annually",
                    "unit": "%",
                },
                "notice_period": {
                    "normal": "1-2 months",
                    "range": "15 days - 3 months",
                    "unit": "days",
                },
                "maintenance_responsibility": {
                    "flag_if_tenant": True,
                    "note": "Tenant should not be responsible for structural repairs",
                },
                "early_termination_penalty": {
                    "typical": "forfeiture of deposit",
                    "suspicious": "multiple months rent",
                },
            },
            "service": {
                "payment_terms": {
                    "normal": "30 days",
                    "range": "15-60 days",
                    "unit": "days",
                },
                "liability_cap": {
                    "normal": "contract value",
                    "suspicious": "unlimited",
                },
                "termination_notice": {
                    "normal": "30 days",
                    "range": "15-90 days",
                    "unit": "days",
                },
                "ip_assignment": {
                    "expected": "work-for-hire or client license",
                    "suspicious": "assignment of contractor pre-existing IP",
                },
                "sla_response_time": {
                    "critical": "<24 hours",
                    "typical": "24-72 hours",
                },
            },
            "nda": {
                "confidentiality_duration_years": {
                    "normal": "1-5",
                    "suspicious": ">5 or perpetual",
                },
                "disclosure_exceptions": {"required": True},
                "return_or_destruction": {"required": True},
            },
            "supplier": {
                "delivery_window_days": {
                    "typical": "30-90",
                    "suspicious": ">180",
                    "unit": "days",
                },
                "warranty_months": {
                    "typical": "3-24",
                    "suspicious": "none",
                    "unit": "months",
                },
                "payment_terms": {"normal": "30-60 days", "unit": "days"},
            },
            "contractor": {
                "milestone_payments": {"recommended": True},
                "retention_percentage": {
                    "typical": "5-10%",
                    "suspicious": ">20%",
                    "unit": "%",
                },
                "insurance_required": {"recommended": True},
            },
            "loan": {
                "interest_rate_max": {
                    "typical_annual_pct": "market_rate",
                    "suspicious_high_pct": ">30%",
                },
                "prepayment_penalty": {"flag_if_present": True},
                "security_description_required": {"required": True},
            },
            "partnership": {
                "profit_sharing_defined": {"required": True},
                "exit_mechanism": {"required": True},
                "voting_rights": {"required": True},
            },
            "software": {
                "license_scope": {"required": True},
                "support_sla": {"recommended": True},
                "source_code_escrow": {"recommended_if_critical": True},
            },
            "construction": {
                "performance_bond": {"typical_pct": "5-10%", "unit": "%"},
                "defect_liability_period_months": {"typical": "6-24", "unit": "months"},
                "liquidated_damages_rate": {
                    "typical": "0.1-0.5% per week",
                    "suspicious": ">1% per week",
                },
            },
            "procurement": {
                "order_of_precedence": {"recommended": True},
                "price_adjustment_clause": {"flag_if_absent": True},
                "termination_for_convenience": {"expected": True},
            },
            "sale_purchase": {
                "transfer_of_title": {"required": True},
                "inspection_period_days": {"typical": "7-30", "unit": "days"},
                "warranty_duration": {"typical_months": "3-12", "unit": "months"},
            },
            # Cross-cutting / general contract benchmarks
            "general": {
                "effective_date_required": {"required": True},
                "governing_law_required": {"required": True},
                "currency_specified": {"required": True},
                "nda_duration": {
                    "normal_years": "1-5",
                    "suspicious": "perpetual or >5 years",
                },
                "unlimited_liability": {"suspicious": True},
                "perpetual_terms": {
                    "suspicious_phrases": ["perpetual", "in perpetuity"]
                },
                "indemnity_scope": {
                    "flag_if_includes_innocent_negligence": True,
                    "note": "Indemnities covering innocent negligence or unlimited IP indemnities are high risk",
                },
                "ambiguous_payment_terms": {
                    "flag_phrases": ["reasonable time", "as agreed", "promptly"]
                },
                "assignment_restrictions": {
                    "flag_if_unrestricted": True,
                    "note": "Assignment without consent can transfer risk",
                },
            },
        }

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def identify_contract_type(self, text: str) -> str:
        """Identify the type of contract"""
        text_lower = text.lower()

        # Expanded keyword lists for more specific contract categories
        keyword_categories = {
            "employment": [
                "employee",
                "employment",
                "salary",
                "designation",
                "probation",
                "resignation",
                "epf",
                "esi",
                "gratuity",
            ],
            "rental": [
                "tenant",
                "landlord",
                "rent",
                "lease",
                "property",
                "premises",
                "security deposit",
                "broker",
            ],
            "service": [
                "service",
                "client",
                "deliverables",
                "milestone",
                "payment terms",
                "sla",
                "support",
            ],
            "nda": [
                "confidential",
                "non-disclosure",
                "nda",
                "confidentiality",
                "proprietary",
                "trade secret",
            ],
            "software": [
                "license",
                "software",
                "source code",
                "escrow",
                "api",
                "saas",
                "hosting",
            ],
            "construction": [
                "construction",
                "contractor",
                "builder",
                "site",
                "drawings",
                "completion",
                "defect liability",
            ],
            "supplier": [
                "supply",
                "supplier",
                "goods",
                "delivery",
                "inspection",
                "warranty",
            ],
            "procurement": [
                "procure",
                "purchase order",
                "tender",
                "bid",
                "po",
                "rfq",
                "rfi",
            ],
            "loan": [
                "loan",
                "interest",
                "principal",
                "repayment",
                "mortgage",
                "security",
                "foreclosure",
            ],
            "partnership": [
                "partner",
                "partnership",
                "profit share",
                "capital contribution",
                "dissolution",
            ],
            "contractor": [
                "independent contractor",
                "subcontractor",
                "retention",
                "milestone",
                "work order",
            ],
            "sale_purchase": [
                "sale",
                "purchase",
                "delivery",
                "goods",
                "transfer of title",
                "inspection",
            ],
        }

        # Score each category by keyword matches
        scores = {cat: 0 for cat in keyword_categories}
        for cat, keywords in keyword_categories.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[cat] += 1

        # Pick the highest-scoring category; tiebreaker by a simple priority list
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_scores[0]

        # Return the top category
        return top_cat

    def get_legal_embeddings(self, text: str):
        """Get embeddings using InLegalBERT"""
        if not self.tokenizer or not self.legal_model:
            return None

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True
            )
            with torch.no_grad():
                outputs = self.legal_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()
        except Exception as e:
            st.error(f"Error getting embeddings: {str(e)}")
            return None

    def create_document_embeddings(
        self, text: str, chunk_size: int = 300, overlap: int = 50
    ):
        """Chunk the document and create embeddings for each chunk using InLegalBERT.

        Args:
            text: Full contract text
            chunk_size: Number of tokens per chunk (reduced to 300 for safety)
            overlap: Number of overlapping tokens between chunks
        """
        if not self.tokenizer or not self.legal_model:
            st.warning("InLegalBERT model not available. Skipping document embedding.")
            self.doc_chunks = []
            self.doc_embeddings = None
            return

        try:
            # Split text into smaller character-based chunks FIRST
            # This prevents tokenization of huge text blocks
            char_chunk_size = 1500  # ~300-400 tokens worth of characters
            text_chunks = []

            for i in range(0, len(text), char_chunk_size - 200):  # 200 char overlap
                chunk = text[i : i + char_chunk_size]
                if chunk.strip():
                    text_chunks.append(chunk)

            st.info(f"Split document into {len(text_chunks)} preliminary chunks")

            # Generate embeddings for each chunk
            embeddings = []
            final_chunks = []

            for i, chunk_text in enumerate(text_chunks):
                try:
                    # Tokenize with IMMEDIATE truncation - this is the critical fix
                    inputs = self.tokenizer(
                        chunk_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding="max_length",
                        add_special_tokens=True,
                    )

                    # Double-check token count
                    actual_length = inputs["input_ids"].shape[1]
                    if actual_length > 512:
                        st.warning(
                            f"Chunk {i} exceeds limit ({actual_length}), skipping..."
                        )
                        continue

                    # Get attention mask
                    attention_mask = inputs.get("attention_mask")

                    # Generate embeddings
                    with torch.no_grad():
                        outputs = self.legal_model(**inputs)

                    # Use attention-weighted mean pooling
                    last_hidden = outputs.last_hidden_state

                    if attention_mask is not None:
                        mask_expanded = (
                            attention_mask.unsqueeze(-1)
                            .expand(last_hidden.size())
                            .float()
                        )
                        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                        pooled = (sum_embeddings / sum_mask).squeeze(0)
                    else:
                        pooled = last_hidden.mean(dim=1).squeeze(0)

                    # Convert to numpy and normalize
                    emb = pooled.cpu().numpy()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm

                    embeddings.append(emb)
                    final_chunks.append(chunk_text)

                except Exception as e:
                    st.warning(f"Error creating embedding for chunk {i}: {e}")
                    continue

            if embeddings:
                self.doc_chunks = final_chunks
                self.doc_embeddings = np.vstack(embeddings)
                st.success(
                    f"✅ Created {len(final_chunks)} document chunks with embeddings"
                )
            else:
                self.doc_chunks = []
                self.doc_embeddings = None
                st.error("❌ Failed to create any embeddings")

        except Exception as e:
            st.error(f"Error in create_document_embeddings: {e}")
            self.doc_chunks = []
            self.doc_embeddings = None

    def semantic_search_chunks(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[int, float, str]]:
        """Retrieve top_k most relevant chunks using cosine similarity.

        Args:
            query: User's question
            top_k: Number of chunks to return

        Returns:
            List of (chunk_index, similarity_score, chunk_text) tuples
        """
        if not hasattr(self, "doc_embeddings") or self.doc_embeddings is None:
            return []

        if len(self.doc_chunks) == 0:
            return []

        try:
            # Truncate query BEFORE tokenization if it's too long
            if len(query) > 1000:
                query = query[:1000]

            # Generate query embedding with IMMEDIATE truncation
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
                add_special_tokens=True,
            )

            # Verify token count
            actual_length = inputs["input_ids"].shape[1]
            if actual_length > 512:
                st.error(
                    f"Query still too long ({actual_length} tokens), cannot process"
                )
                return []

            attention_mask = inputs.get("attention_mask")

            with torch.no_grad():
                outputs = self.legal_model(**inputs)

            last_hidden = outputs.last_hidden_state

            # Apply attention-weighted pooling
            if attention_mask is not None:
                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                query_emb = (sum_embeddings / sum_mask).squeeze(0)
            else:
                query_emb = last_hidden.mean(dim=1).squeeze(0)

            # Normalize query embedding
            query_emb = query_emb.cpu().numpy()
            query_norm = np.linalg.norm(query_emb)
            if query_norm > 0:
                query_emb = query_emb / query_norm

            # Compute cosine similarity with all document chunks
            similarities = self.doc_embeddings @ query_emb

            # Get top_k indices
            top_indices = np.argsort(-similarities)[:top_k]

            # Return results
            results = []
            for idx in top_indices:
                results.append(
                    (int(idx), float(similarities[idx]), self.doc_chunks[idx])
                )

            return results

        except Exception as e:
            st.error(f"Error in semantic search: {e}")
            return []

    def chat_with_document(
        self,
        prompt: str,
        chat_history: List[Dict],
        model: str = "phi-3.5-mini-instruct",
        max_response_tokens: int = 500,  
        temperature: float = 0.3,
    ) -> str:
        """RAG-based chat with contract document using InLegalBERT retrieval + LM Studio local LLM.

        Args:
            prompt: User's question
            chat_history: Previous conversation messages
            model: Model name (ignored for local LM Studio)
            max_response_tokens: Max tokens in response
            temperature: Response randomness (0-1)

        Returns:
            Generated response string
        """
        import requests

        # LM Studio local endpoint
        LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

        # Ensure document embeddings exist
        if not hasattr(self, "doc_chunks") or not self.doc_chunks:
            if hasattr(self, "last_full_text") and self.last_full_text:
                with st.spinner("Creating document embeddings..."):
                    self.create_document_embeddings(self.last_full_text)
            else:
                return "⚠️ No contract document loaded. Please upload a contract first."

        # RETRIEVAL: Get relevant chunks using semantic search
        with st.spinner("🔍 Searching relevant contract sections..."):
            relevant_chunks = self.semantic_search_chunks(prompt, top_k=2)  # Reduced from 3 to 2

        if not relevant_chunks:
            return "⚠️ Could not retrieve relevant information from the contract. Please try rephrasing your question."

        # Build context from retrieved chunks with length limits
        context_parts = []
        total_context_chars = 0
        max_context_chars = 1000  # Reduced from 2000 to 1000
        
        for idx, score, chunk_text in relevant_chunks:
            # Only include chunks with high similarity (>0.5)
            if score > 0.5:  # Increased from 0.3 to 0.5
                # Truncate chunk if too long
                if len(chunk_text) > 400:  # Reduced from 800 to 400
                    chunk_text = chunk_text[:400] + "..."
                
                # Check if adding this chunk would exceed limit
                if total_context_chars + len(chunk_text) > max_context_chars:
                    break
                    
                context_parts.append(chunk_text)  # Removed relevance score
                total_context_chars += len(chunk_text)

        context = "\n\n".join(context_parts)  # Removed separator

        # GENERATION: Create prompt for LM Studio
        system_prompt = """You are a legal assistant for Indian contracts. Answer based ONLY on provided context. Be brief (3-4 sentences max). Don't provide legal advice."""

        user_prompt = f"""Context: {context}

    Question: {prompt}"""

        # Call LM Studio local API (no chat history for faster response)
        try:
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}  # Removed conversation_context
                ],
                "max_tokens": max_response_tokens,
                "temperature": temperature,
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=1000)
            
            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"❌ LM Studio API error: {response.status_code} - {response.text}"

            # Add source attribution
            answer += f"\n\n*[Based on {len(context_parts)} contract sections]*"

            return answer

        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to LM Studio. Please ensure:\n1. LM Studio is running\n2. Server is started (green button in LM Studio)\n3. A model is loaded\n4. Server is running on port 1234"
        except requests.exceptions.Timeout:
            return "⚠️ LM Studio request timed out. The model might be processing. Try a shorter question."
        except Exception as e:
            error_str = str(e).lower()
            
            if "connection" in error_str:
                return "❌ Cannot connect to LM Studio. Make sure it's running on http://localhost:1234"
            else:
                return f"❌ Error calling LM Studio API: {str(e)}"


    def analyze_legal_clauses(self, text: str) -> List[Dict]:
        """Analyze and extract key legal clauses"""
        clauses: List[Dict] = []

        # Helper to extract numeric metadata from clause text
        def extract_clause_meta(clause_text: str) -> Dict:
            meta: Dict = {
                "months": [],
                "days": [],
                "percent": [],
                "amounts": [],
                "dates": [],
            }

            # months/days
            for m in re.findall(
                r"(\d+)\s*(?:months|month|mos|m)\b", clause_text, re.IGNORECASE
            ):
                try:
                    meta["months"].append(int(m))
                except:
                    pass
            for d in re.findall(r"(\d+)\s*(?:days|day)\b", clause_text, re.IGNORECASE):
                try:
                    meta["days"].append(int(d))
                except:
                    pass

            # percentages
            for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", clause_text):
                try:
                    meta["percent"].append(float(p))
                except:
                    pass

            # currency amounts (₹, Rs, INR, numeric with commas)
            for amt in re.findall(
                r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)", clause_text, re.IGNORECASE
            ):
                try:
                    cleaned = amt.replace(",", "")
                    meta["amounts"].append(float(cleaned))
                except:
                    pass

            # ISO-like dates and simple date phrases (e.g., 1 January 2020)
            for dt in re.findall(r"(\d{1,2}\s+\w+\s+\d{4})", clause_text):
                meta["dates"].append(dt)

            return meta

        # Expanded patterns to capture more clause types
        patterns = {
            "termination": r"terminat[ie].*?(?:\.|;|\n)",
            "termination_for_convenience": r"termination for convenience.*?(?:\.|;|\n)",
            "non_compete": r"non.?compet.*?(?:\.|;|\n)",
            "confidentiality": r"confidential.*?(?:\.|;|\n)",
            "nda": r"non-?disclos.*?(?:\.|;|\n)|\bnda\b.*?(?:\.|;|\n)",
            "payment": r"payment.*?(?:\.|;|\n)",
            "payment_terms": r"payment terms.*?(?:\.|;|\n)",
            "liability": r"liabilit.*?(?:\.|;|\n)",
            "indemnity": r"indemnit.*?(?:\.|;|\n)",
            "notice": r"notice.*?(?:\.|;|\n)",
            "penalty": r"penalt.*?(?:\.|;|\n)",
            "governing_law": r"governing law|jurisdiction|seat of arbitration.*?(?:\.|;|\n)",
            "effective_date": r"effective date.*?(?:\.|;|\n)|effective from.*?(?:\.|;|\n)",
            "security_deposit": r"security deposit.*?(?:\.|;|\n)",
            "probation": r"probation.*?(?:\.|;|\n)",
            "severance": r"severance.*?(?:\.|;|\n)",
            "ip": r"intellectual property|invention|assign.*?ip|ownership of deliverables.*?(?:\.|;|\n)",
            "sla": r"sla|service level|uptime|response time|availability.*?(?:\.|;|\n)",
            "warranty": r"warrant.*?(?:\.|;|\n)",
            "retention": r"retention.*?(?:\.|;|\n)",
            "performance_bond": r"performance bond|bank guarantee.*?(?:\.|;|\n)",
            "assignment": r"assign.*?(?:\.|;|\n)",
            "taxes": r"gst|tax(es)?|tds|withhold.*?(?:\.|;|\n)",
            "stamp_duty": r"stamp duty|registration.*?(?:\.|;|\n)",
            "arbitration": r"arbitrat.*?(?:\.|;|\n)",
        }

        # Search and extract
        for clause_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            # Limit number of matches per clause type to avoid large outputs
            for match in matches[:5]:
                clause_text = match.strip()
                meta = extract_clause_meta(clause_text)
                risk = self.assess_clause_risk(clause_type, clause_text)

                # Additional heuristic risk upgrades based on benchmarks
                # e.g., perpetual NDA language
                if clause_type in ("nda", "confidentiality"):
                    if re.search(
                        r"perpetual|in perpetuity", clause_text, re.IGNORECASE
                    ):
                        risk = "high"

                clauses.append(
                    {
                        "type": clause_type,
                        "text": clause_text,
                        "meta": meta,
                        "risk_level": risk,
                    }
                )

        return clauses

    def assess_clause_risk(self, clause_type: str, clause_text: str) -> str:
        """Assess risk level of a clause"""
        high_risk_keywords = [
            "unlimited",
            "perpetual",
            "irrevocable",
            "sole discretion",
            "waive all rights",
            "forfeit",
            "unconditional",
            "absolute",
            "entire",
            "maximum",
        ]

        medium_risk_keywords = [
            "reasonable",
            "material breach",
            "substantial",
            "significant",
            "binding",
            "exclusive",
            "restrict",
            "prohibit",
        ]

        clause_lower = clause_text.lower()

        if any(keyword in clause_lower for keyword in high_risk_keywords):
            return "high"
        elif any(keyword in clause_lower for keyword in medium_risk_keywords):
            return "medium"
        else:
            return "low"

    def generate_plain_language_explanation(
        self, clauses: List[Dict], contract_type: str
    ) -> List[Dict]:
        """Generate plain language explanations for clauses"""
        explanations = []

        for clause in clauses:
            explanation = {
                "original": clause["text"],
                "plain_english": self.convert_to_plain_language(
                    clause["text"], clause["type"]
                ),
                "practical_meaning": self.get_practical_meaning(
                    clause["text"], clause["type"], contract_type
                ),
                "risk_level": clause["risk_level"],
            }
            explanations.append(explanation)

        return explanations

    def convert_to_plain_language(self, clause: str, clause_type: str) -> str:
        """Convert legal jargon to plain language using LM Studio"""
        import requests
        
        LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
        
        if len(clause) > 500:
            clause = clause[:500] + "..."
        
        try:
            payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "Convert legal text to simple English. Keep it brief (2-3 sentences)."},
                    {"role": "user", "content": f"Rewrite in plain language:\n\n{clause}"}
                ],
                "max_tokens": 150,
                "temperature": 0.3,
            }
            
            response = requests.post(LM_STUDIO_URL, json=payload, timeout=120)
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"Error: Status {response.status_code}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    def get_practical_meaning(
        self, clause: str, clause_type: str, contract_type: str
    ) -> str:
        """Get practical meaning of clauses"""
        # Expanded template-based practical meanings for many clause types
        templates = {
            "termination": {
                "employment": "Explains how and when employment can be ended, notice required, and consequences for termination.",
                "rental": "Explains how the tenancy can be ended by tenant or landlord and what notices are required.",
                "service": "Explains termination rights for both parties and any transition assistance required.",
            },
            "termination_for_convenience": {
                "default": "Allows a party to end the contract without cause, usually with notice and sometimes with compensation."
            },
            "non_compete": {
                "employment": "Restricts post-employment work in certain roles, industries or geographies — affects future job options.",
                "service": "May restrict the contractor from doing similar work for others — check scope and duration.",
            },
            "confidentiality": {
                "default": "Defines what information must be kept confidential, duration of confidentiality and permitted disclosures."
            },
            "nda": {
                "default": "A separate NDA or confidentiality section: sets limits on use and disclosure of sensitive information."
            },
            "payment": {
                "employment": "Describes salary, payroll cycle, and deductions (taxes, benefits).",
                "rental": "Specifies rent due dates, late fees and payment methods.",
                "service": "Specifies invoicing, due dates, and accepted payment methods.",
            },
            "payment_terms": {
                "default": "Describes when invoices are due, any milestones, and consequences of late payment."
            },
            "liability": {
                "default": "Defines financial responsibility for loss; look for caps, exclusions and insurance requirements."
            },
            "indemnity": {
                "default": "Obligation to compensate for third-party claims — watch for scope (IP, negligence) and caps."
            },
            "notice": {
                "default": "How to give formal notices (method, address, and timelines) which trigger contractual processes."
            },
            "penalty": {
                "default": "Penalty clauses impose remedies or charges for breaches — ensure they're proportional and defined."
            },
            "governing_law": {
                "default": "Specifies which country's/state's law governs the contract and where disputes will be resolved."
            },
            "effective_date": {
                "default": "Specifies when contract obligations start — important for timelines, limitations and notice periods."
            },
            "security_deposit": {
                "rental": "An amount held to secure performance — check amount, interest, and refund conditions.",
                "default": "Funds held as security for obligations; check refund and use conditions.",
            },
            "probation": {
                "employment": "Initial period of employment when termination may be easier and benefits/entitlements may be limited."
            },
            "severance": {
                "employment": "Compensation on termination — check amount, triggers and calculation method."
            },
            "ip": {
                "default": "Describes ownership and licensing of intellectual property created or used under the contract."
            },
            "sla": {
                "service": "Service levels the supplier must meet (uptime, response time); remedies are often credits rather than direct damages."
            },
            "warranty": {
                "default": "Promises about product/service quality and the remedies available if warranties are breached."
            },
            "retention": {
                "contractor": "A portion of payment held back until completion/defects liability period to ensure performance."
            },
            "performance_bond": {
                "construction": "Security (bond/bank guarantee) to ensure contractor completes work; check claim conditions and expiry."
            },
            "assignment": {
                "default": "Whether parties can transfer rights/obligations to third parties — unrestricted assignment can transfer risk."
            },
            "taxes": {
                "default": "Specifies who bears GST/TDS/withholding and any gross-up obligations."
            },
            "stamp_duty": {
                "default": "Indicates whether stamp duty/registration is required which can affect enforceability or additional costs."
            },
            "arbitration": {
                "default": "Specifies arbitration as dispute resolution — check seat, rules and enforceability."
            },
        }

        # Return a contract-type-specific template if available, otherwise default
        entry = templates.get(clause_type, {})
        return entry.get(
            contract_type,
            entry.get(
                "default",
                "This clause defines specific terms and conditions for this agreement.",
            ),
        )

    def detect_red_flags(
        self, clauses: List[Dict], contract_type: str, full_text: str = None
    ) -> List[Dict]:
        """Detect red flags in the contract.

        Uses clause-level risk plus some cross-cutting checks on the full contract text
        when available (e.g., missing governing law, missing effective date, currency).
        """
        red_flags: List[Dict] = []

        # Clause-level red flags
        for clause in clauses:
            if clause.get("risk_level") == "high":
                issue = self.identify_red_flag_issue(clause, contract_type)
                red_flag = {
                    "clause": clause.get("text", "")[:200] + "...",
                    "issue": issue,
                    "severity": "High",
                    "recommendation": self.get_red_flag_recommendation(
                        clause, contract_type
                    ),
                }
                red_flags.append(red_flag)

        # Cross-cutting checks using full_text
        if full_text:
            text_lower = full_text.lower()
            clause_types = {c.get("type") for c in clauses}

            # Governing law / jurisdiction
            if (
                "governing_law" not in clause_types
                and "jurisdiction" not in clause_types
                and not re.search(
                    r"governing law|jurisdiction|seat of arbitration", text_lower
                )
            ):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Governing law / jurisdiction missing",
                        "severity": "High",
                        "recommendation": "Specify governing law and forum (e.g., courts/arbitration seat in India)",
                    }
                )

            # Effective date
            if "effective_date" not in clause_types and not re.search(
                r"effective date|effective from", text_lower
            ):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Effective date not specified",
                        "severity": "Medium",
                        "recommendation": "Add an explicit effective date to avoid ambiguity",
                    }
                )

            # Currency
            if not re.search(r"₹|inr|rs\.?\s|rupee", text_lower):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Currency not specified",
                        "severity": "Medium",
                        "recommendation": "Specify the currency for payments (e.g., INR, ₹)",
                    }
                )

            # Unlimited liability mention anywhere
            if re.search(
                r"unlimited liability|liability shall be unlimited", text_lower
            ):
                red_flags.append(
                    {
                        "clause": "",
                        "issue": "Unlimited liability exposure",
                        "severity": "High",
                        "recommendation": "Seek to cap liability to contract value or insurance limits",
                    }
                )

        return red_flags

    def identify_red_flag_issue(self, clause: Dict, contract_type: str) -> str:
        """Identify specific issues in red flag clauses using clause meta and benchmarks.

        clause: Dict should contain keys 'type', 'text', 'meta'.
        """
        clause_type = clause.get("type", "")
        text = clause.get("text", "").lower()
        meta = clause.get("meta", {}) or {}

        # Helper to parse numeric benchmark values (e.g., '24 months' -> 24)
        def _parse_first_int(s: str):
            if not s:
                return None
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else None

        # Use benchmarks when available
        bench = (
            self.benchmarks.get(contract_type, {})
            if hasattr(self, "benchmarks")
            else {}
        )

        # Specific clause handlers
        if clause_type == "non_compete":
            months = meta.get("months", [])
            if months:
                max_months = None
                # check bench fallback to employment.non_compete.max_acceptable
                if "non_compete" in bench:
                    max_months = _parse_first_int(
                        bench["non_compete"].get("max_acceptable", "")
                    )
                elif (
                    "employment" in self.benchmarks
                    and "non_compete" in self.benchmarks["employment"]
                ):
                    max_months = _parse_first_int(
                        self.benchmarks["employment"]["non_compete"].get(
                            "max_acceptable", ""
                        )
                    )
                if max_months and max(months) > max_months:
                    return f"Overly broad non-compete restrictions ({max(months)} months — exceeds {max_months} months)"
            return "Overly broad non-compete restrictions"

        if clause_type in ("penalty", "bond_penalty"):
            # Check percentages or amounts
            percents = meta.get("percent", [])
            amounts = meta.get("amounts", [])
            if percents and max(percents) > 15:
                return f"Excessive penalty rate ({max(percents)}%)"
            if amounts and any(a > 1e6 for a in amounts):
                return "Excessive monetary penalty amounts"
            return "Excessive penalty amounts"

        if clause_type == "liability":
            if "unlimited" in text:
                return "Unlimited liability exposure"
            return "Potentially broad liability clause"

        if clause_type in ("nda", "confidentiality"):
            if re.search(r"perpetual|in perpetuity", text):
                return "Perpetual confidentiality obligation (NDA)"
            years = meta.get("months", [])
            # months->years heuristic
            if years and max(years) >= 60:
                return "Very long confidentiality duration (>5 years)"
            return "Confidentiality / NDA clause"

        if clause_type == "indemnity":
            if re.search(r"innocent negligence|negligence", text):
                return "Indemnity covering negligence (including potentially innocent negligence)"
            return "Broad indemnity clause"

        if clause_type == "governing_law":
            return "Governing law / jurisdiction specified"

        if clause_type == "effective_date":
            return "Effective date specified"

        if clause_type == "security_deposit":
            # Check if deposit months exceed suspicious value
            months = meta.get("months", [])
            if months and max(months) > 6:
                return f"High security deposit detected ({max(months)} months)"
            return "Security deposit clause"

        # Default
        default_issues = {
            "non_compete": "Overly broad non-compete restrictions",
            "penalty": "Excessive penalty amounts",
            "liability": "Unlimited liability exposure",
            "termination": "Unfair termination conditions",
        }
        return default_issues.get(clause_type, "Potentially unfavorable terms")

    def get_red_flag_recommendation(self, clause: Dict, contract_type: str) -> str:
        """Get targeted recommendations for a red-flag clause using clause meta and benchmarks.

        clause should be a dict with keys 'type', 'text', and optional 'meta'.
        """
        clause_type = clause.get("type", "")
        text = clause.get("text", "")
        meta = clause.get("meta", {}) or {}

        # Helpers
        def _first_meta(key: str):
            vals = meta.get(key, [])
            return vals[0] if vals else None

        # Try to use benchmarks for contract_type, fall back to general employment values
        bench = (
            self.benchmarks.get(contract_type, {})
            if hasattr(self, "benchmarks")
            else {}
        )
        employment_bench = (
            self.benchmarks.get("employment", {}) if hasattr(self, "benchmarks") else {}
        )

        # Recommendations tailored by clause type
        if clause_type == "non_compete":
            months = _first_meta("months")
            max_allowed = None
            if "non_compete" in bench:
                max_allowed = bench["non_compete"].get("max_acceptable")
            elif "non_compete" in employment_bench:
                max_allowed = employment_bench["non_compete"].get("max_acceptable")
            rec = "Negotiate for reasonable time limits and geographic restrictions"
            if months and max_allowed:
                rec += f" — detected {months} months; consider reducing to <= {max_allowed}"
            rec += ". Consider compensation or narrower scope (role/geography)."
            return rec

        if clause_type in ("penalty", "bond_penalty"):
            perc = _first_meta("percent")
            if perc and perc > 15:
                return f"Request clarification and cap on penalty rate ({perc}%). Seek proportional penalties and a clear formula."
            amt = _first_meta("amounts")
            if amt:
                return "Request justification for penalty amounts and seek proportionality to actual damages. Consider an upper cap tied to contract value."
            return "Request clarification on penalty calculation and reasonableness. Consider capping penalties."

        if clause_type == "liability":
            return "Seek a liability cap (e.g., contract value or specified sum) and purchase/confirm appropriate insurance. Exclude indirect/consequential damages."

        if clause_type in ("nda", "confidentiality"):
            years = _first_meta("months")
            rec = "Limit confidentiality duration (e.g., 1-5 years) and add clear carve-outs for pre-existing information and legal disclosures."
            if years and years >= 60:
                rec = "Confidentiality appears very long — negotiate a limited duration (1-5 years) or specify narrow scope."
            if re.search(r"perpetual|in perpetuity", text, re.IGNORECASE):
                rec = "Perpetual NDA detected — negotiate a fixed duration and explicit carve-outs for pre-existing and independently developed information."
            return rec

        if clause_type == "indemnity":
            return "Narrow indemnity obligations: limit to breach, wilful misconduct or third-party IP infringement; exclude innocent negligence where appropriate; cap indemnity exposure."

        if clause_type == "security_deposit":
            months = _first_meta("months")
            rec = "Clarify security deposit amount, interest (if any), and refund conditions. Require reasonableness and timelines for refund."
            if months and months > 6:
                rec = f"Security deposit seems high ({months} months) — negotiate down or request interest and a cap (typical 2-3 months)."
            return rec

        if clause_type == "sla":
            return "Define SLA metrics clearly, include remedies like service credits (not unlimited damages), and cap total liability for SLA breaches. Ensure measurement and reporting are defined."

        if clause_type == "retention":
            return "Negotiate reasonable retention (typical 5-10%) and a defined release schedule after acceptance or defects liability period."

        if clause_type == "performance_bond":
            return "Request a reasonable performance security (e.g., 5-10% of contract) and specify expiry/claim conditions. Consider bank guarantee instead of upfront cash."

        if clause_type == "governing_law":
            return "Ensure governing law and dispute resolution mechanism are clearly defined; prefer arbitration seat in a mutually agreed city in India if suitable."

        if clause_type == "effective_date":
            return (
                "Add an explicit effective date and clarify when obligations commence."
            )

        if clause_type == "taxes":
            return "Specify who bears GST and withholding tax implications (TDS); include gross-up clause if required for certain payments."

        if clause_type == "assignment":
            return "Restrict assignment without consent, or require notice and conditions for assignment to protect parties from unexpected transfers."

        # Default recommendation
        return "Seek legal review for this clause and negotiate for clearer limits, caps or carve-outs as appropriate."

    def generate_checklist(self, contract_type: str, clauses: List[Dict]) -> List[str]:
        """Generate personalized checklist

        Supports multiple contract categories. Returns an ordered, deduplicated
        list of checklist items combining category-specific items, India- specific
        general checks, and items derived from detected clauses.
        """
        checklist: List[str] = []

        # Base checklist items by contract type (expanded categories)
        base_checklists: Dict[str, List[str]] = {
            "employment": [
                "Verify salary amount, components (basic, HRA, allowances) and in-hand pay",
                "Confirm EPF/ESI applicability and employer contributions",
                "Check gratuity, bonus/variable pay and calculation method",
                "Understand probation period, confirmation criteria and termination during probation",
                "Check notice period length and pay-in-lieu rules",
                "Review leave policies (paid leave, sick leave, maternity/paternity) and encashment",
                "Confirm working hours, overtime rules and compensation",
                "Review non-compete, non-solicit and restrictive covenants for scope and duration",
                "Check IP and invention assignment clauses (pre-existing contractor IP should be carved out)",
                "Verify performance targets/KPIs and appraisal frequency",
                "Confirm relocation/transfer and travel expense reimbursement policies",
                "Ensure statutory compliance (Shops & Establishment, labour laws) is referenced",
                "Confirm tax/TDS treatment and whether gross-up applies for certain benefits",
                "Look for background-check or offer-conditional clauses and their limits",
            ],
            "rental": [
                "Confirm rent amount, payment frequency and accepted payment methods",
                "Check security deposit amount, interest (if any) and refund conditions",
                "Verify whether stamp duty/registration is required (e.g., >11 months in many states)",
                "Clarify responsibility for maintenance, repairs and major structural works",
                "Check rent escalation clause and maximum cap or formula",
                "Confirm notice period for termination by landlord/tenant and early termination penalties",
                "Verify subletting and assignment restrictions",
                "Ensure inventory/condition report is attached or described",
                "Confirm who pays municipal/property taxes and utilities",
                "Check use-of-premises clause (residential vs commercial) and required approvals",
                "Look for broker fee responsibility and any other prepaid charges",
                "Check renewal terms and whether rent at renewal is capped or negotiable",
            ],
            "service": [
                "Clarify detailed scope, deliverables, acceptance criteria and milestone definitions",
                "Verify payment schedule, invoice requirements and GST handling",
                "Check late payment interest rate and dispute resolution on invoices",
                "Confirm IP ownership or license scope; carve out contractor pre-existing IP",
                "Review confidentiality and data protection obligations (PII handling, cross-border transfer)",
                "Check liability cap, exclusions (consequential/indirect) and indemnity carve-outs",
                "Verify SLA metrics, remedies, service credits and caps on remedies",
                "Confirm insurance requirements (professional indemnity, public liability)",
                "Check subcontracting/assignment rules and approval process",
                "Confirm termination for convenience vs for cause and transition assistance obligations",
                "Review tax clauses: GST applicability, who bears taxes, and TDS provisions",
                "Check governing law, dispute resolution and seat of arbitration (city in India)",
                "Ensure change control process and pricing for out-of-scope work is defined",
                "Check warranty period, support obligations and defect remediation timelines",
            ],
            "nda": [
                "Confirm parties and defined confidential information",
                "Check duration of confidentiality and whether perpetual NDA is used",
                "Verify permitted disclosures (court order, affiliates) and notice requirements",
                "Confirm return/destruction obligations for confidential materials",
                "Ensure carve-outs for pre-existing and independently developed information",
            ],
            "supplier": [
                "Check delivery terms, incoterms (if international) and delivery timelines",
                "Verify quality acceptance criteria and inspection rights",
                "Ensure clear pricing, taxes (GST) and invoice documentation",
                "Confirm warranty and remedy for defective goods/services",
                "Check lead times and penalties for late delivery",
            ],
            "contractor": [
                "Confirm contractor status (independent contractor vs employee) and liabilities",
                "Check milestones, deliverables and acceptance testing",
                "Verify IP assignment vs license and pre-existing IP carve-outs",
                "Review indemnity and insurance requirements",
                "Check payment milestones and retention terms",
            ],
            "loan": [
                "Confirm principal amount, interest rate (fixed/variable) and repayment schedule",
                "Check default interest, grace periods and events of default",
                "Verify security/collateral descriptions and perfection steps (hypothecation, mortgage)",
                "Confirm prepayment, foreclosure charges and break costs",
                "Review representations, covenants and cross-default clauses",
            ],
            "partnership": [
                "Verify capital contributions, ownership percentages and profit sharing",
                "Check management voting rights and decision-making processes",
                "Confirm exit/transfer rules, buy-sell mechanisms and valuation methods",
                "Review non-compete and non-solicit obligations between partners",
                "Ensure dispute resolution and dissolution processes are defined",
            ],
            "software": [
                "Confirm licensed rights (scope, duration, users) and restrictions",
                "Check support and maintenance obligations, SLAs and response times",
                "Verify escrow and source code access on vendor insolvency (if critical)",
                "Review data protection, hosting location and cross-border transfer terms",
                "Ensure clear warranty, limitation of liability and IP indemnity clauses",
            ],
            "construction": [
                "Verify scope, drawings, specifications and approval process",
                "Check performance bonds, retention and defect liability period",
                "Confirm milestone payments, variations and change orders handling",
                "Review health & safety, site access and local statutory compliance",
                "Ensure delay/liquidated damages clauses are reasonable and capped",
            ],
            "procurement": [
                "Confirm tender/PO references, acceptance and order of precedence",
                "Verify supplier qualification criteria and documentation requirements",
                "Check price adjustment, indexation and escalation clauses",
                "Ensure termination for convenience with defined notice and compensation",
                "Review audit and compliance rights for suppliers",
            ],
            "sale_purchase": [
                "Confirm description of goods/services, quantity and delivery terms",
                "Check transfer of title and risk allocation",
                "Verify warranties, returns and indemnities for breach of specifications",
                "Confirm pricing, taxes (GST) and payment terms",
                "Ensure conditions precedent (inspections, approvals) are clearly stated",
            ],
        }

        checklist.extend(base_checklists.get(contract_type, []))
        if not base_checklists.get(contract_type):
            # For unknown contract types, include a small generic starter
            checklist.append("Confirm scope, parties, effective date and payment terms")

        # Add specific items based on detected clauses
        for clause in clauses:
            if clause.get("risk_level") in ["high", "medium"]:
                if "non_compete" in clause.get("type", ""):
                    checklist.append(
                        "Ask for clarification on non-compete restrictions and duration"
                    )
                elif "penalty" in clause.get("type", ""):
                    checklist.append("Request justification for penalty amounts")
                elif "confidentiality" in clause.get("type", ""):
                    checklist.append(
                        "Understand what information is considered confidential"
                    )

        # Preserve order and remove duplicates
        return list(dict.fromkeys(checklist))

    def calculate_confidence_score(
        self, clauses: List[Dict], red_flags: List[Dict]
    ) -> Dict:
        """Calculate overall confidence score"""
        total_clauses = len(clauses)
        high_risk_clauses = sum(1 for c in clauses if c["risk_level"] == "high")
        medium_risk_clauses = sum(1 for c in clauses if c["risk_level"] == "medium")

        # Calculate score (0-100)
        if total_clauses == 0:
            score = 50
        else:
            penalty = (high_risk_clauses * 20) + (medium_risk_clauses * 10)
            score = max(0, 100 - penalty)

        # Determine rating
        if score >= 80:
            rating = "Excellent"
            color = "#4CAF50"
        elif score >= 60:
            rating = "Good"
            color = "#FFC107"
        elif score >= 40:
            rating = "Fair"
            color = "#FF9800"
        else:
            rating = "Poor"
            color = "#F44336"

        return {
            "score": score,
            "rating": rating,
            "color": color,
            "details": {
                "total_clauses": total_clauses,
                "high_risk": high_risk_clauses,
                "medium_risk": medium_risk_clauses,
                "red_flags": len(red_flags),
            },
        }

    def suggest_alternatives(
        self, red_flags: List[Dict], contract_type: str
    ) -> List[Dict]:
        """Suggest alternative clause wordings"""
        alternatives = []

        for flag in red_flags:
            alternative = {
                "original_issue": flag["issue"],
                "suggested_wording": self.get_alternative_wording(flag, contract_type),
                "explanation": "This alternative provides more balanced terms",
            }
            alternatives.append(alternative)

        return alternatives

    def get_alternative_wording(self, red_flag: Dict, contract_type: str) -> str:
        """Get alternative clause wording"""
        alternatives = {
            "Overly broad non-compete restrictions": "Employee agrees not to work for direct competitors in the same role within [specific geographic area] for [reasonable time period] months.",
            "Excessive penalty amounts": "Any penalties shall be reasonable and proportionate to actual damages incurred.",
            "Unlimited liability exposure": "Liability shall be limited to the total value of this contract or actual damages, whichever is lower.",
            "Unfair termination conditions": "Either party may terminate this agreement with [reasonable notice period] days written notice.",
        }

        return alternatives.get(
            red_flag["issue"], "Consider negotiating for more balanced and fair terms."
        )


def main():
    st.markdown(
        """
    <div class="main-header">
        <h1>🏛️ Indian Contract Analyzer</h1>
        <p>Simplify legal contracts with AI-powered analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize the analyzer
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = ContractAnalyzer()

    # Ensure contract_text persists
    if "contract_text" not in st.session_state:
        st.session_state.contract_text = None

    # Ensure chat_history persists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    st.sidebar.header("📋 Navigation")
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["📄 Full Contract Analysis", "💬 Chat with Document", "📊 Bulk Analysis"],
    )

    if analysis_mode == "📄 Full Contract Analysis":
        show_full_analysis()
    elif analysis_mode == "💬 Chat with Document":
        show_chat_interface()
    elif analysis_mode == "📊 Bulk Analysis":
        show_bulk_analysis()


def show_full_analysis():
    """Show the full contract analysis interface"""
    st.header("📄 Upload Your Contract")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload your Indian contract in PDF format",
    )

    contract_text = None

    if uploaded_file is not None:
        with st.spinner("📖 Reading and analyzing your contract..."):
            # Extract text
            contract_text = st.session_state.analyzer.extract_text_from_pdf(
                uploaded_file
            )

            if contract_text:
                # Store in session state
                st.session_state.contract_text = contract_text
                st.session_state.contract_filename = getattr(
                    uploaded_file, "name", "Unknown"
                )

                # Store full text on analyzer for RAG
                st.session_state.analyzer.last_full_text = contract_text

                # Create document embeddings for semantic search
                try:
                    with st.spinner(
                        "🔗 Creating document embeddings for semantic search..."
                    ):
                        st.session_state.analyzer.create_document_embeddings(
                            contract_text
                        )
                    st.success("🔎 Document embeddings ready for chat/search.")
                except Exception as e:
                    st.warning(
                        f"Could not create document embeddings: {str(e)}. Chat will work in simple mode."
                    )

                # Compute analysis
                st.session_state.contract_type = (
                    st.session_state.analyzer.identify_contract_type(contract_text)
                )
                st.session_state.clauses = (
                    st.session_state.analyzer.analyze_legal_clauses(contract_text)
                )
            else:
                st.error(
                    "❌ Could not extract text from the PDF. Please try a different file."
                )

    # If no new upload, check for previously stored contract text
    if (
        contract_text is None
        and "contract_text" in st.session_state
        and st.session_state.contract_text
    ):
        contract_text = st.session_state.contract_text
        if (
            "contract_filename" in st.session_state
            and st.session_state.contract_filename
        ):
            st.info(
                f"Using previously uploaded contract: {st.session_state.contract_filename}"
            )

    # Show analysis if we have contract text
    if (
        contract_text
        and "contract_type" in st.session_state
        and "clauses" in st.session_state
    ):
        contract_type = st.session_state.contract_type
        clauses = st.session_state.clauses

        # Show analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🔍 Plain Language",
                "🚨 Red Flags",
                "✅ Checklist",
                "📊 Confidence Score",
                "💡 Alternatives",
            ]
        )

        with tab1:
            show_plain_language_analysis(clauses, contract_type)

        with tab2:
            show_red_flag_analysis(clauses, contract_type)

        with tab3:
            show_checklist(clauses, contract_type)

        with tab4:
            show_confidence_score(clauses)

        with tab5:
            show_alternatives(clauses, contract_type)
    else:
        st.info("Upload a PDF contract to begin analysis.")


def show_plain_language_analysis(clauses, contract_type):
    """Show plain language explanations"""
    st.header("🔍 Plain Language Explanations")

    if not clauses:
        st.warning("No specific legal clauses detected in the contract.")
        return

    explanations = st.session_state.analyzer.generate_plain_language_explanation(
        clauses, contract_type
    )

    st.info(f"📋 **Contract Type Detected:** {contract_type.title()}")

    # Show only the first instance of each clause type
    seen_types = set()
    shown_count = 0
    for idx, clause in enumerate(clauses):
        ctype = clause.get("type", "unknown")
        if ctype in seen_types:
            continue
        seen_types.add(ctype)
        shown_count += 1

        # Get the matching explanation
        if idx < len(explanations):
            explanation = explanations[idx]
        else:
            explanation = {
                "original": clause.get("text", ""),
                "plain_english": st.session_state.analyzer.convert_to_plain_language(
                    clause.get("text", ""), ctype
                ),
                "practical_meaning": st.session_state.analyzer.get_practical_meaning(
                    clause.get("text", ""), ctype, contract_type
                ),
                "risk_level": clause.get("risk_level", "medium"),
            }

        with st.expander(
            f"Clause {shown_count}: {ctype.replace('_', ' ').title()}", expanded=True
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📜 Original Text")
                st.text_area(
                    "Original",
                    explanation.get("original", ""),
                    height=100,
                    key=f"original_{ctype}_{shown_count}",
                    label_visibility="collapsed",
                )

            with col2:
                st.subheader("✨ Plain Language")
                st.text_area(
                    "Plain",
                    explanation.get("plain_english", ""),
                    height=100,
                    key=f"plain_{ctype}_{shown_count}",
                    label_visibility="collapsed",
                )

            st.subheader("🎯 What This Means")
            risk_class = f"risk-{explanation.get('risk_level', 'medium')}"

            practical = explanation.get("practical_meaning", "")
            # Append detected metadata
            meta = clause.get("meta") or {}
            if meta:
                meta_parts = []
                for k, v in meta.items():
                    if v:  # Only show non-empty values
                        meta_parts.append(f"{k}: {v}")
                if meta_parts:
                    practical = f"{practical}<br><small>Detected: {', '.join(meta_parts)}</small>"

            st.markdown(
                f"""
            <div class="{risk_class}">
                <strong>Practical Impact:</strong> {practical}<br>
                <strong>Risk Level:</strong> {explanation.get('risk_level', 'medium').upper()}
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_red_flag_analysis(clauses, contract_type):
    """Show red flag detection"""
    st.header("🚨 Red Flag Detection")

    # Pass full contract text for comprehensive red flag detection
    full_text = st.session_state.get("contract_text", "")
    red_flags = st.session_state.analyzer.detect_red_flags(
        clauses, contract_type, full_text=full_text
    )

    if not red_flags:
        st.success("🎉 No major red flags detected in this contract!")
        return

    st.warning(f"⚠️ Found {len(red_flags)} potential red flags:")

    for i, flag in enumerate(red_flags, 1):
        with st.expander(f"🚨 Red Flag {i}: {flag['issue']}", expanded=True):
            st.markdown(
                f"""
            <div class="risk-high">
                <strong>Issue:</strong> {flag["issue"]}<br>
                <strong>Severity:</strong> {flag["severity"]}<br>
                <strong>Recommendation:</strong> {flag["recommendation"]}
            </div>
            """,
                unsafe_allow_html=True,
            )

            if flag.get("clause"):
                st.text_area(
                    "Problematic Clause:",
                    flag["clause"],
                    height=100,
                    key=f"redflag_{i}",
                    label_visibility="visible",
                )


def show_checklist(clauses, contract_type):
    """Show personalized checklist"""
    st.header("✅ Before You Sign Checklist")

    checklist = st.session_state.analyzer.generate_checklist(contract_type, clauses)

    st.info("📝 **Important questions to ask before signing:**")

    for i, item in enumerate(checklist, 1):
        checked = st.checkbox(f"{item}", key=f"checklist_{i}")
        if checked:
            st.success("✅ Completed")


def show_confidence_score(clauses):
    """Show confidence score and metrics"""
    st.header("📊 Contract Confidence Score")

    full_text = st.session_state.get("contract_text", "")
    red_flags = st.session_state.analyzer.detect_red_flags(
        clauses, st.session_state.contract_type, full_text=full_text
    )
    confidence = st.session_state.analyzer.calculate_confidence_score(
        clauses, red_flags
    )

    # Score visualization
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence["score"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Contract Safety Score"},
            delta={"reference": 80},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": confidence["color"]},
                "steps": [
                    {"range": [0, 40], "color": "#ffebee"},
                    {"range": [40, 60], "color": "#fff3e0"},
                    {"range": [60, 80], "color": "#f3e5f5"},
                    {"range": [80, 100], "color": "#e8f5e8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Score details
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Rating", confidence["rating"])

    with col2:
        st.metric("Total Clauses", confidence["details"]["total_clauses"])

    with col3:
        st.metric("High Risk", confidence["details"]["high_risk"])

    with col4:
        st.metric("Red Flags", confidence["details"]["red_flags"])

    # Recommendations
    st.subheader("📋 Recommendations")

    if confidence["score"] >= 80:
        st.success("✅ This contract appears to be fair and well-structured.")
    elif confidence["score"] >= 60:
        st.warning(
            "⚠️ This contract has some areas of concern. Review the red flags carefully."
        )
    else:
        st.error(
            "❌ This contract has significant issues. Consider legal review before signing."
        )


def show_alternatives(clauses, contract_type):
    """Show alternative clause suggestions"""
    st.header("💡 Suggested Alternatives")

    full_text = st.session_state.get("contract_text", "")
    red_flags = st.session_state.analyzer.detect_red_flags(
        clauses, contract_type, full_text=full_text
    )

    if not red_flags:
        st.success("🎉 No alternatives needed - your contract looks good!")
        return

    alternatives = st.session_state.analyzer.suggest_alternatives(
        red_flags, contract_type
    )

    st.info("💡 **Suggested improvements for problematic clauses:**")

    for i, alt in enumerate(alternatives, 1):
        with st.expander(f"Alternative {i}: {alt['original_issue']}", expanded=True):
            st.subheader("⚠️ Current Issue")
            st.error(alt["original_issue"])

            st.subheader("💡 Suggested Improvement")
            st.success(alt["suggested_wording"])

            st.subheader("📝 Why This Is Better")
            st.info(alt["explanation"])


def show_chat_interface():
    """Show enhanced chat interface for document interaction"""
    st.header("💬 Chat with Your Contract")

    # Check if contract is loaded
    if "contract_text" not in st.session_state or not st.session_state.contract_text:
        st.warning(
            "📤 Please upload and analyze a contract first in the 'Full Contract Analysis' tab!"
        )
        return

    # Display contract info
    if "contract_type" in st.session_state:
        st.info(f"📋 Contract Type: **{st.session_state.contract_type.title()}**")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history with better formatting
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(
                    f"""
                <div style="background-color: #1565c0; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🧑 You:</strong><br>{message['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div style="background-color: #42a5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🤖 Assistant:</strong><br>{message['content']}
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Suggested questions
    if not st.session_state.chat_history:
        st.subheader("💡 Suggested Questions:")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📋 What are the key terms?"):
                suggested_q = "What are the key terms and conditions in this contract?"
                st.session_state.pending_question = suggested_q
                st.rerun()

            if st.button("⚠️ What are the risks?"):
                suggested_q = "What are the potential risks or unfavorable clauses in this contract?"
                st.session_state.pending_question = suggested_q
                st.rerun()

        with col2:
            if st.button("💰 Payment terms?"):
                suggested_q = "What are the payment terms and conditions?"
                st.session_state.pending_question = suggested_q
                st.rerun()

            if st.button("🚪 Termination clauses?"):
                suggested_q = "What are the termination conditions and notice periods?"
                st.session_state.pending_question = suggested_q
                st.rerun()

    # Chat input
    st.markdown("---")
    user_question = st.text_input(
        "Ask a question about your contract:",
        key="chat_input",
        placeholder="e.g., What is the notice period for termination?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send 📤", type="primary")
    with col2:
        if st.button("Clear Chat 🗑️"):
            st.session_state.chat_history = []
            if "pending_question" in st.session_state:
                del st.session_state.pending_question
            st.rerun()

    # Handle pending question from suggested buttons
    if "pending_question" in st.session_state:
        user_question = st.session_state.pending_question
        del st.session_state.pending_question
        send_button = True

    # Process user input
    if send_button and user_question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Generate response using RAG
        with st.spinner("🤔 Analyzing contract and generating response..."):
            response = st.session_state.analyzer.chat_with_document(
                user_question, st.session_state.chat_history
            )

        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.rerun()


def generate_chat_response(question: str, contract_text: str) -> str:
    """Generate response to user questions (simplified version)"""
    question_lower = question.lower()

    # Simple keyword-based responses (replace with GPT API in production)
    if "salary" in question_lower or "pay" in question_lower:
        return "I can see payment-related clauses in your contract. Let me help you understand the salary and payment terms..."
    elif "termination" in question_lower or "quit" in question_lower:
        return "Regarding termination clauses, your contract specifies certain conditions for ending the employment..."
    elif "notice" in question_lower:
        return "The notice period requirements in your contract are important to understand..."
    else:
        return "I can help you understand any specific clause in your contract. Could you be more specific about which section you're asking about?"


def show_bulk_analysis():
    """Show bulk analysis interface"""
    st.header("📊 Bulk Contract Analysis")
    st.info("Upload multiple contracts for comparative analysis")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload multiple contracts for comparison",
    )

    if uploaded_files:
        if st.button("Analyze All Contracts"):
            results = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Analyzing {file.name}..."):
                    text = st.session_state.analyzer.extract_text_from_pdf(file)
                    if text:
                        contract_type = (
                            st.session_state.analyzer.identify_contract_type(text)
                        )
                        clauses = st.session_state.analyzer.analyze_legal_clauses(text)
                        red_flags = st.session_state.analyzer.detect_red_flags(
                            clauses, contract_type
                        )
                        confidence = (
                            st.session_state.analyzer.calculate_confidence_score(
                                clauses, red_flags
                            )
                        )

                        results.append(
                            {
                                "filename": file.name,
                                "type": contract_type,
                                "score": confidence["score"],
                                "rating": confidence["rating"],
                                "red_flags": len(red_flags),
                                "clauses": len(clauses),
                            }
                        )

                progress_bar.progress((i + 1) / len(uploaded_files))

            # Display results
            if results:
                df = pd.DataFrame(results)
                st.subheader("📊 Analysis Results")
                st.dataframe(df)

                # Visualization
                fig = px.bar(
                    df,
                    x="filename",
                    y="score",
                    title="Contract Safety Scores Comparison",
                    color="rating",
                    color_discrete_map={
                        "Excellent": "#4CAF50",
                        "Good": "#FFC107",
                        "Fair": "#FF9800",
                        "Poor": "#F44336",
                    },
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Score", f"{df['score'].mean():.1f}")
                with col2:
                    st.metric("Best Contract", df.loc[df["score"].idxmax(), "filename"])
                with col3:
                    st.metric("Total Red Flags", df["red_flags"].sum())


if __name__ == "__main__":
    main()
