"""
🎬🎵 AI Music & Film Recommendation System
Landing Page - User chooses between Music or Film recommendation
"""

import streamlit as st

# Page configuration - COLLAPSED sidebar for clean look
st.set_page_config(
    page_title="AI Recommendation System",
    page_icon="🎬",
    layout="centered",  # Centered for minimalist look
    initial_sidebar_state="collapsed"  # NO SIDEBAR on landing page!
)

# Custom CSS - Clean professional design
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Full screen background - Professional slate */
    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }

    /* Container */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    /* Hero section */
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #CBD5E1;
        margin-bottom: 3rem;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: #6366F1;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
    }

    /* Style buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.75rem 2rem;
        border-radius: 500px;
        border: none;
        transition: all 0.2s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4);
    }

    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
    }

    .feature-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 1rem;
    }

    .feature-description {
        font-size: 0.95rem;
        color: #CBD5E1;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        padding: 3rem 0 1rem 0;
        margin-top: 4rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">AI Recommendation System</div>
    <div class="hero-subtitle">Discover your next favorite music or film</div>
</div>
""", unsafe_allow_html=True)

# Two-column layout for cards
col1, col2 = st.columns(2, gap="large")

with col1:
    # Music Card
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎵</div>
        <div class="feature-title">Music</div>
        <div class="feature-description">
            Discover songs based on your mood<br>
            • Mood-based filtering<br>
            • 114,000+ songs<br>
            • Interactive analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Explore Music", key="music_btn", use_container_width=True):
        st.switch_page("pages/1_Music.py")

with col2:
    # Film Card
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎬</div>
        <div class="feature-title">Film</div>
        <div class="feature-description">
            Find your next favorite movie<br>
            • Smart filters<br>
            • 7,400+ films<br>
            • Platform suggestions
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Explore Film", key="film_btn", use_container_width=True):
        st.switch_page("pages/2_Film.py")

# Footer
st.markdown("""
<div class="footer">
    Final Project Kelompok 4 • AI-Powered Recommendations
</div>
""", unsafe_allow_html=True)
