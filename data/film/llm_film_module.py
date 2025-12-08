"""
Film LLM Chatbot Module
Exported from Agent_Film notebook for use in Streamlit app

This module contains the chatbot logic using LangChain + Gemini AI
"""

import os
import json
import re
import pandas as pd
from typing import Dict, Any, List
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FilmLLMChatbot:
    """
    Film recommendation chatbot using Gemini 2.0 Flash
    """

    def __init__(self, film_df, tfidf_matrix=None, cosine_sim=None, api_key=None):
        """
        Initialize chatbot with film data and similarity matrices

        Args:
            film_df: DataFrame with film data
            tfidf_matrix: Pre-computed TF-IDF matrix (optional, will build if None)
            cosine_sim: Pre-computed cosine similarity matrix (optional, will build if None)
            api_key: Google API key (optional, can use env var)
        """
        self.film_df = film_df
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.llm = None
        self.agent = None
        self.chat_history = []
        self.last_query = ""

        # Build or use pre-computed matrices
        if tfidf_matrix is None or cosine_sim is None:
            self.vectorizer, self.tfidf_matrix, self.cosine_sim = self._build_similarity_matrices()
        else:
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
            self.tfidf_matrix = tfidf_matrix
            self.cosine_sim = cosine_sim

        # Build title index
        self.film_df["title_clean"] = (
            self.film_df["title"]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]", "", regex=True)
            .str.strip()
        )
        self.indices = pd.Series(self.film_df.index, index=self.film_df["title_clean"]).drop_duplicates()

        # System prompt (from notebook cell-27)
        self.system_prompt = """
Kamu adalah chatbot khusus FILM.
Hanya jawab pertanyaan seputar film dalam dataset.
Jika user bertanya di luar film: TOLAK dengan sopan.

Tugasmu:
- Jika user menyebut judul film apapun yang ada di dataset:
   → SELALU panggil tool search_movie(judul)
   → Tampilkan info lengkap dengan format:
        🎬 Judul:
        📖 Deskripsi:
        🎭 Genre:
        ⭐ Rating:
        🎬 Sutradara:
        👥 Aktor:
        ⏳ Durasi:
        📅 Tahun:
- Jika user meminta "mirip", "similar", "rekomendasi", "yang seperti ...":
   → Setelah memanggil search_movie(judul),
     WAJIB panggil recommend_movie(judul)
   → Tampilkan daftar rekomendasi
- Jika user bertanya tentang tahun, aktor, sutradara, genre, rating → WAJIB panggil tool search_free
- Jika user menjawab "boleh", "lanjut", "oke", "iya" → Berikan info lanjutan
- Semua jawaban WAJIB dalam bahasa Indonesia

ATURAN FORMAT:
- Gunakan line-break dan format multiline
- Tampilkan emoji untuk visual appeal

CATATAN:
- Untuk pertanyaan yang mengandung judul film, WAJIB tampilkan info film lengkap terlebih dahulu
"""

        self.non_film_keywords = ["presiden", "politik", "agama", "integral", "anjing", "kucing", "cuaca"]

        # Initialize if API key available
        if self.api_key:
            self._initialize_llm()

    def _build_similarity_matrices(self):
        """Build TF-IDF and cosine similarity matrices"""
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),
            max_features=50000
        )

        # Create soup (description + actors + directors + genres)
        def clean_text(x):
            if isinstance(x, str):
                return re.sub(r"[^a-zA-Z0-9\s]", " ", x.lower()).strip()
            return ""

        soup = (
            self.film_df["description"].apply(clean_text) + " " +
            self.film_df["actors"].astype(str).str.lower().str.replace(" ", "_") + " " +
            self.film_df["directors"].astype(str).str.lower().str.replace(" ", "_") + " " +
            self.film_df["genres_list"].astype(str).str.lower().str.replace(" ", "_")
        )

        tfidf_matrix = vectorizer.fit_transform(soup)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        return vectorizer, tfidf_matrix, cosine_sim

    def _initialize_llm(self):
        """Initialize LLM and agent"""
        try:
            # Initialize Gemini 2.0 Flash
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                api_key=self.api_key
            )

            # Create tools
            tools = self._create_tools()

            # Build agent
            self.agent = initialize_agent(
                tools=tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True
            )

            return True
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            return False

    def _create_tools(self):
        """Create LangChain tools for the chatbot"""

        @tool
        def search_movie(title: str):
            """Mencari detail film berdasarkan judul."""
            clean = re.sub(r"[^a-z0-9]", "", str(title).lower()).strip()

            # Exact match
            if clean in self.indices:
                idx = self.indices[clean]
            else:
                # Fuzzy fallback
                matches = [k for k in self.indices.index if clean in k]
                if matches:
                    idx = self.indices[matches[0]]
                else:
                    return {"error": f"Film '{title}' tidak ditemukan."}

            row = self.film_df.iloc[idx]

            return {
                "Detail film": row.get("title"),
                "Deskripsi": row.get("description"),
                "Tahun Rilis": row.get("release_year"),
                "Genre": row.get("genres_list"),
                "Rating": row.get("rating"),
                "Sutradara": row.get("directors"),
                "Aktor": row.get("actors"),
                "Durasi": row.get("runtime_minutes")
            }

        @tool
        def recommend_movie(title: str):
            """Memberi rekomendasi film mirip berdasarkan judul."""
            t = re.sub(r"[^a-z0-9]", "", title.lower()).strip()

            if t not in self.indices:
                return {"error": f"Film '{title}' tidak ditemukan."}

            idx = self.indices[t]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

            rec = []
            for i, score in sim_scores:
                row = self.film_df.iloc[i]
                rec.append({
                    "Judul": row.get("title"),
                    "Tahun": row.get("release_year"),
                    "Genre": row.get("genres_list"),
                    "Rating": row.get("rating"),
                    "Durasi": row.get("runtime_minutes"),
                    "Similarity": float(score)
                })

            return {"recommendations": rec}

        @tool
        def search_free(query: str = ""):
            """
            Pencarian bebas: rating tertinggi/terendah, aktor, sutradara, genre, tahun
            """
            q = str(query).lower().strip()

            # Rating tertinggi
            if "rating tertinggi" in q or "rating tinggi" in q or "paling bagus" in q:
                self.film_df["rating_num"] = pd.to_numeric(self.film_df["rating"], errors="coerce")
                hasil = self.film_df.sort_values("rating_num", ascending=False).head(5)
                return hasil.to_dict(orient="records")

            # Rating terendah
            if "rating terendah" in q or "rating rendah" in q:
                self.film_df["rating_num"] = pd.to_numeric(self.film_df["rating"], errors="coerce")
                hasil = self.film_df.sort_values("rating_num", ascending=True).head(5)
                return hasil.to_dict(orient="records")

            # Genre
            genres = ["action", "horror", "drama", "comedy", "thriller", "romance"]
            for g in genres:
                if g in q:
                    subset = self.film_df[self.film_df["genres_list"].astype(str).str.lower().str.contains(g)]
                    if not subset.empty:
                        subset["rating_num"] = pd.to_numeric(subset["rating"], errors="coerce")
                        return subset.sort_values("rating_num", ascending=False).head(5).to_dict(orient="records")

            # Tahun
            year_match = re.search(r"\b(19|20)\d{2}\b", q)
            if year_match:
                yr = int(year_match.group(0))
                subset = self.film_df[self.film_df["release_year"].astype(int) == yr]
                if not subset.empty:
                    return subset.head(10).to_dict(orient="records")

            # Judul contains query
            subset = self.film_df[self.film_df["title"].astype(str).str.lower().str.contains(q)]
            if not subset.empty:
                return subset.head(5).to_dict(orient="records")

            return [{"error": "Tidak ada film yang cocok dengan query."}]

        return [search_movie, recommend_movie, search_free]

    def _retrieve_context(self, question, top_k=3):
        """RAG retrieval for context"""
        try:
            q_vec = self.vectorizer.transform([question])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            top_idx = sims.argsort()[::-1][:top_k]

            ctx = []
            for i in top_idx:
                row = self.film_df.iloc[i]
                ctx.append(f"Judul: {row.get('title')} | Genre: {row.get('genres_list')} | Rating: {row.get('rating')}")
            return "\n".join(ctx)
        except:
            return ""

    def is_film_related(self, text: str) -> bool:
        """Check if query is film-related"""
        if any(keyword in text.lower() for keyword in self.non_film_keywords):
            return False
        return True

    def chat(self, user_message: str, thread_id: str = "default") -> str:
        """
        Main chat function

        Args:
            user_message: User's message
            thread_id: Thread ID for conversation (default: "default")

        Returns:
            Bot's response
        """
        if not self.llm or not self.agent:
            return "Error: Chatbot belum diinisialisasi. Pastikan GOOGLE_API_KEY sudah diset."

        # Check if film-related
        if not self.is_film_related(user_message):
            return "Maaf, saya hanya dapat membantu rekomendasi film. Coba tanya tentang film yuk! 🎬"

        # Handle context continuation (boleh, lanjut, etc)
        if user_message.lower().strip() in ["boleh", "bolehh", "ya", "iya", "lanjut", "oke", "y"]:
            if self.last_query:
                user_message = self.last_query
            else:
                return "Silakan tanyakan tentang film yang ingin Anda cari! 🎬"

        self.last_query = user_message

        try:
            # Get context
            context = self._retrieve_context(user_message)

            # Build query
            final_query = f"{self.system_prompt}\n\nKonteks:\n{context}\n\nUser: {user_message}\nJawab:"

            # Invoke agent
            result = self.agent.invoke(final_query)

            # Extract response
            if isinstance(result, dict):
                response = result.get("output") or result.get("output_text") or str(result)
            else:
                response = str(result)

            return response

        except Exception as e:
            return f"Maaf, terjadi error: {str(e)}"

    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        self.last_query = ""


# Convenience function for easy import
def create_chatbot(film_df, tfidf_matrix=None, cosine_sim=None, api_key=None):
    """
    Create a film chatbot instance

    Args:
        film_df: DataFrame with film data
        tfidf_matrix: Pre-computed TF-IDF matrix (optional)
        cosine_sim: Pre-computed cosine similarity matrix (optional)
        api_key: Google API key (optional)

    Returns:
        FilmLLMChatbot instance
    """
    return FilmLLMChatbot(film_df, tfidf_matrix, cosine_sim, api_key)
