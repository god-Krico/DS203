import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def setup_semantic_search():
    """Initialize the semantic search model and load data."""
    if 'model' not in st.session_state:
        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    if 'df' not in st.session_state:
        try:
            st.session_state.df = pd.read_csv("reclustered_ranked_sessions.csv")
            st.session_state.df["embedding"] = st.session_state.df["Cleaned_Summary"].apply(
                lambda x: st.session_state.model.encode(str(x), convert_to_numpy=True)
            )
            
            # Compute cluster embeddings
            st.session_state.cluster_embeddings = (
                st.session_state.df.groupby("Reclustered")["embedding"]
                .apply(lambda x: np.mean(np.vstack(x), axis=0))
                .to_dict()
            )
        except FileNotFoundError:
            st.error("Could not find the required CSV file. Please ensure 'reclustered_ranked_sessions.csv' is in the correct directory.")
            return False
    return True

def get_semantic_results(query):
    """Get semantic search results for the query."""
    query_embedding = st.session_state.model.encode(query, convert_to_numpy=True)
    
    # Compute similarities
    similarities = {
        cluster: cosine_similarity([query_embedding], [embedding])[0][0]
        for cluster, embedding in st.session_state.cluster_embeddings.items()
    }
    
    # Get best cluster
    best_cluster = max(similarities, key=similarities.get)
    
    # Get top summaries
    top_summaries = (
        st.session_state.df[st.session_state.df["Reclustered"] == best_cluster]
        .sort_values("Rank")
        .head(3)
    )
    
    return best_cluster, similarities[best_cluster], top_summaries

# Set page config with custom theme
st.set_page_config(
    page_title="DS203: Lecture Summary Analyzer",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode with futuristic blue tones
st.markdown("""
    <style>
    /* Import Open Sans font */
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600&display=swap');
    
    /* Apply font to all elements */
    * {
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Main background */
    .stApp {
        background-color: #0a1929;
    }
    
    /* Text input styling */
    .stTextInput>div>div>input {
        background-color: #1a2b3c;
        color: #00ffff;
        border: 1px solid #1e4b8c;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1e4b8c;
        color: #00ffff;
        border: none;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #00ffff !important;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    p, div {
        color: #ccd6f6 !important;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #1a2b3c;
        color: #00ffff !important;
        border: 1px solid #1e4b8c;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #1a2b3c;
        color: #00ffff !important;
        border: 1px solid #1e4b8c;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a1929;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #1e4b8c;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00ffff;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 100%;
        background-color: #0a1929;
        padding: 8px;
        text-align: center;
        border-top: 1px solid #1e4b8c;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .footer p {
        margin: 0;
        padding: 2px;
        font-size: 0.9em;
        text-align: center;
        width: 100%;
        font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
    }
    
    .footer-content {
        max-width: 100%;
        margin: 0 auto;
        padding: 0 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with custom styling and increased font sizes
st.markdown("<h1 style='text-align: center; color: #00ffff !important; font-family: Open Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, Open Sans, Helvetica Neue, sans-serif !important; font-size: 2.2em;'>📚 DS203: Lecture Summary Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #1e4b8c !important; font-family: Open Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, Open Sans, Helvetica Neue, sans-serif !important; font-size: 1.3em;'>Recover Lost Associations using the Power Of Data!</h3>", unsafe_allow_html=True)

# Initialize semantic search
if setup_semantic_search():
    # Search bar with improved styling
    search_query = st.text_input(
        "Enter keywords to find relevant lecture sessions:",
        placeholder="e.g., machine learning, neural networks, data visualization...",
        key="search_input"
    )

    # Display results based on search query
    if search_query:
        with st.spinner("🔍 Searching for relevant sessions..."):
            best_cluster, similarity_score, top_summaries = get_semantic_results(search_query)
            
            # Enhanced display of the most relevant cluster
            st.markdown(f"""
            <div style='background-color: #1a2b3c; padding: 12px; border-radius: 8px; border: 1px solid #00ffff; margin-bottom: 15px;'>
                <h3 style='color: #00ffff; text-align: center; margin-bottom: 10px; font-size: 1.1em;'>🎯 Most Relevant Session Cluster</h3>
                <div style='display: flex; justify-content: center; align-items: center; gap: 15px;'>
                    <div style='background-color: #0a1929; padding: 8px; border-radius: 6px; border: 1px solid #1e4b8c;'>
                        <p style='color: #00ffff; margin: 0; font-size: 1em;'>Cluster {best_cluster}</p>
                    </div>
                    <div style='background-color: #0a1929; padding: 8px; border-radius: 6px; border: 1px solid #1e4b8c;'>
                        <p style='color: #00ffff; margin: 0; font-size: 1em;'>Relevance Score: {similarity_score:.4f}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📝 Top 3 Relevant Summaries")
            for _, row in top_summaries.iterrows():
                with st.expander(f"Rank {row['Rank']} - Score: {similarity_score:.4f}"):
                    st.markdown(f"""
                    <div style='background-color: #1a2b3c; padding: 15px; border-radius: 5px; border: 1px solid #1e4b8c;'>
                        {row['Session_Summary']}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("👆 Enter keywords to discover relevant lecture sessions and their summaries.")

# Footer with team information
st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <p style='color: #00ffff; font-size: 1.1em;'>Made by:</p>
            <p style='color: #ccd6f6; font-size: 1em; white-space: nowrap;'>
                Rushabh Bonde - 23B0703 | Karan Satarkar - 23B0708 | Pratik Jadhao - 23B0719 | Ashwin Mankar - 23B0726
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)