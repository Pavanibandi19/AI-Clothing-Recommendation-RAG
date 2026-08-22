import sys
import html
import logging
from pathlib import Path
import streamlit as st

# Configure quiet third-party logging to prevent console lag and noisy traces
logging.basicConfig(level=logging.INFO)
for logger_name in (
    "chromadb", "chromadb.telemetry", "sentence_transformers", 
    "transformers", "urllib3", "httpx", "opentelemetry"
):
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Add both project root and src directory to sys.path for robust local execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.config import (
    VALID_CATEGORIES, VALID_GENDERS, VALID_FITS, 
    VALID_MATERIALS, VALID_COLORS, DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
)
from src.retrieval import ProductRetriever
from src.agent import RAGAgent


def safe_text(value):
    return html.escape(str(value or ""))


def render_product_card(column, product):
    """Renders a single clothing product card in the specified Streamlit column."""
    with column:
        brand = safe_text(product.get("brand", "Unknown Brand"))
        name = safe_text(product.get("product_name", "Untitled Product"))
        desc = safe_text(product.get("description", "No description available."))
        cat = safe_text(product.get("category", "Unknown")).title()
        gender = safe_text(product.get("gender", "Unknown"))
        color = safe_text(product.get("color", "Unknown"))
        fit = safe_text(product.get("fit", "Unknown"))
        material = safe_text(product.get("material", "Unknown"))
        season = safe_text(product.get("season", "Unknown"))
        rating = safe_text(product.get("rating", 0))
        price = int(product.get("price_inr", 0))
        match = safe_text(product.get("match_percentage", 90))

        img_url = product.get("image_url")
        if img_url:
            st.image(img_url, use_container_width=True)

        st.markdown(f"**{brand}** — {name}")
        st.write(f"**₹{price}**  •  ⭐ {rating}  •  **{match}% Match**")
        st.write(f"Category: {cat} | Gender: {gender} | Color: {color} | Fit: {fit} | Material: {material} | Season: {season}")
        st.write(desc)

        prod_url = product.get("product_url") or product.get("product_link") or product.get("url") or product.get("website_url")
        if prod_url:
            st.link_button("🔗 View Product", str(prod_url))


# Cache expensive system initialization
@st.cache_resource(show_spinner="Initializing AI Clothing Retrieval Engine...")
def get_retriever():
    return ProductRetriever()


@st.cache_resource(show_spinner="Initializing Ollama AI Agent...")
def get_agent():
    return RAGAgent()


def run_app():
    # Page configuration
    st.set_page_config(
        page_title="AI Clothing Recommendation System",
        page_icon="👔",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    retriever = get_retriever()
    agent = get_agent()

    # Initialize session state variables
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""

    if "search_submitted" not in st.session_state:
        st.session_state["search_submitted"] = False

    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []

    # Sidebar Filters
    st.sidebar.markdown("### 🧥👗 AI CLOTHING")
    st.sidebar.markdown("Filter catalog constraints directly or let the AI extract them from your search.")

    selected_gender = st.sidebar.selectbox("Gender", ["All"] + VALID_GENDERS, index=0, key="sidebar_gender")
    selected_cat = st.sidebar.selectbox("Category", ["All"] + VALID_CATEGORIES, index=0, key="sidebar_category")
    selected_color = st.sidebar.selectbox("Color", ["All"] + sorted(VALID_COLORS), index=0, key="sidebar_color")
    selected_fit = st.sidebar.selectbox("Fit", ["All"] + VALID_FITS, index=0, key="sidebar_fit")
    selected_material = st.sidebar.selectbox("Material", ["All"] + VALID_MATERIALS, index=0, key="sidebar_material")

    price_range = st.sidebar.slider(
        "Price Range (₹)",
        min_value=0,
        max_value=6000,
        value=(0, 6000),
        step=100,
        key="sidebar_price_range"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Engine Settings")
    top_k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=DEFAULT_TOP_K, key="sidebar_top_k")
    sim_threshold = st.sidebar.slider(
        "Semantic Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=DEFAULT_SIMILARITY_THRESHOLD, 
        step=0.05,
        key="sidebar_sim_threshold"
    )

    # Search History in Sidebar
    if st.session_state["search_history"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🕒 Recent Searches")
        for past_q in reversed(st.session_state["search_history"][-5:]):
            st.sidebar.caption(f"• {past_q}")

    # Main Header
    st.title("👔 AI Clothing Recommendation System")
    st.markdown("Discover curated fashion powered by **Natural Language Understanding**, **Hard Attribute Constraints**, and **ChromaDB Vector Retrieval**.")

    # 1. Popular Searches Section
    st.markdown("##### 💡 Popular Searches")
    pop_col1, pop_col2, pop_col3, pop_col4 = st.columns(4)

    if pop_col1.button("👖 Blue slim fit jeans under ₹2000", use_container_width=True, key="popular_search_jeans"):
        st.session_state["user_query"] = "blue slim fit jeans under 2000"
        st.session_state["clothing_search_input"] = "blue slim fit jeans under 2000"
        st.session_state["search_submitted"] = True

    if pop_col2.button("👕 Black casual t-shirts below ₹1500", use_container_width=True, key="popular_search_tshirts"):
        st.session_state["user_query"] = "black casual t-shirts below 1500"
        st.session_state["clothing_search_input"] = "black casual t-shirts below 1500"
        st.session_state["search_submitted"] = True

    if pop_col3.button("👗 Women's cotton dresses for summer", use_container_width=True, key="popular_search_dresses"):
        st.session_state["user_query"] = "women's cotton dresses for summer"
        st.session_state["clothing_search_input"] = "women's cotton dresses for summer"
        st.session_state["search_submitted"] = True

    if pop_col4.button("🧥 Men's oversized hoodies under ₹2500", use_container_width=True, key="popular_search_hoodies"):
        st.session_state["user_query"] = "men's oversized hoodies under 2500"
        st.session_state["clothing_search_input"] = "men's oversized hoodies under 2500"
        st.session_state["search_submitted"] = True

    # 2. Main Search Input Section
    entered_text = st.text_input(
        "Enter your style request, budget, fabric, or occasion:",
        value=st.session_state.get("user_query", ""),
        placeholder="e.g. black casual t-shirts below 1500, women's cotton dresses for summer...",
        key="clothing_search_input"
    )

    if st.button("🔍 Find Outfits", use_container_width=True, key="btn_find_outfits"):
        st.session_state["user_query"] = entered_text
        st.session_state["search_submitted"] = True

    # Current active user query (strictly separated from AI responses)
    active_user_query = st.session_state.get("user_query", "").strip()

    # 3. Search Execution Pipeline
    if st.session_state.get("search_submitted") and active_user_query:
        if not retriever.query_parser.is_clothing_query(active_user_query):
            st.warning("⚠️ Please enter a clothing-related search, such as 'blue jeans under 2000' or 'cotton shirts for men'.")
            st.stop()

        # Track in search history
        if active_user_query not in st.session_state["search_history"]:
            st.session_state["search_history"].append(active_user_query)

        # Build UI filters dictionary
        ui_filter_dict = {
            "gender": None if selected_gender == "All" else selected_gender,
            "category": None if selected_cat == "All" else selected_cat,
            "brand": None,
            "color": None if selected_color == "All" else selected_color,
            "fit": None if selected_fit == "All" else selected_fit,
            "material": None if selected_material == "All" else selected_material,
            "min_price": price_range[0] if price_range[0] > 0 else None,
            "max_price": price_range[1] if price_range[1] < 6000 else None,
        }

        try:
            with st.spinner("🔍 Filtering dataset & running vector similarity search..."):
                retrieval_results = retriever.retrieve(
                    query=active_user_query,
                    ui_filters=ui_filter_dict,
                    top_k=top_k,
                    similarity_threshold=sim_threshold
                )
        except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
            st.error("⚠️ Retrieval failed while processing your query.")
            st.exception(exc)
            st.stop()

        parsed_constraints = retrieval_results.get("parsed_query", {})
        recommended_products = retrieval_results.get("items", [])

        # Display Detected Filter Badges
        st.markdown("#### 🎯 Active Parsed Constraints")
        badge_labels = []
        if parsed_constraints.get("max_price"):
            badge_labels.append(f"Max Price: ₹{int(parsed_constraints['max_price'])}")
        if parsed_constraints.get("min_price"):
            badge_labels.append(f"Min Price: ₹{int(parsed_constraints['min_price'])}")
        if parsed_constraints.get("gender"):
            badge_labels.append(f"Gender: {parsed_constraints['gender']}")
        if parsed_constraints.get("category"):
            badge_labels.append(f"Category: {parsed_constraints['category']}")
        if parsed_constraints.get("color"):
            badge_labels.append(f"Color: {parsed_constraints['color']}")
        if parsed_constraints.get("fit"):
            badge_labels.append(f"Fit: {parsed_constraints['fit']}")
        if parsed_constraints.get("material"):
            badge_labels.append(f"Material: {parsed_constraints['material']}")

        if badge_labels:
            st.write(" • ".join(badge_labels))
        else:
            st.caption("No explicit hard metadata constraints detected in text; performing global semantic search.")

        st.write("")

        if not recommended_products:
            st.error("⚠️ **No matching products found**")
            st.info(
                "No clothing items in the dataset satisfied all your hard metadata constraints (such as max price, gender, or category). "
                "Try relaxing sidebar filters or increasing your price limit."
            )
        else:
            # 4. DISPLAY PRODUCTS FIRST (Immediate rendering, zero delay)
            st.markdown(f"### 🛍️ Recommended For You ({len(recommended_products)} Matches)")

            # Display Products Grid (2 columns per row)
            for i in range(0, len(recommended_products), 2):
                grid_col1, grid_col2 = st.columns(2)
                render_product_card(grid_col1, recommended_products[i])
                if i + 1 < len(recommended_products):
                    render_product_card(grid_col2, recommended_products[i + 1])

            st.write("")

            # 5. AI STYLIST RATIONALE (Rendered strictly BELOW products)
            st.markdown("### 🤖 AI Stylist Rationale")
            try:
                with st.spinner("Consulting AI Stylist..."):
                    explanation = agent.generate_recommendation_explanation(
                        query=active_user_query,
                        recommended_products=recommended_products,
                        parsed_query=parsed_constraints
                    )
                explanation_text = explanation.get("text", "").strip()
                explanation_source = explanation.get("source", "Fashion Assistant").strip()
                if not explanation_text:
                    fallback = agent._generate_fallback_summary(active_user_query, recommended_products, parsed_constraints)
                    explanation_text = fallback["text"]
                    explanation_source = fallback["source"]

                with st.container():
                    st.markdown(explanation_text)
                    st.caption(f"🧠 Rationale Source: {explanation_source}")
            except Exception as exc:
                fallback = agent._generate_fallback_summary(active_user_query, recommended_products, parsed_constraints)
                with st.container():
                    st.markdown(fallback["text"])
                    st.caption(f"🧠 Rationale Source: {fallback['source']}")

            # 6. RAG Retrieval Diagnostics Expander (Optional Developer Panel)
            diagnostics = retrieval_results.get("diagnostics", {})
            with st.expander("🛠️ Developer / RAG Diagnostics (Optional)"):
                col_diag1, col_diag2, col_diag3 = st.columns(3)
                v_active = diagnostics.get("vector_search_active", False)
                col_diag1.metric("Vector Search Mode", "ChromaDB Dense Vector" if v_active else "Metadata Fallback")
                col_diag2.metric("Chroma Index Size", f"{diagnostics.get('chroma_count', 0)} products")
                col_diag3.metric("Hard Filter Candidates", f"{diagnostics.get('candidates_count', len(recommended_products))} products")

                st.markdown("##### 📐 Per-Product Relevance Scores & Vector Metrics")
                diag_rows = []
                d_map = diagnostics.get("distances_map", {})
                for p in recommended_products:
                    pid = str(p.get("product_id", ""))
                    raw_dist = d_map.get(pid, "1.0000 (unranked fallback)")
                    dist_str = f"{raw_dist:.4f}" if isinstance(raw_dist, float) else str(raw_dist)
                    diag_rows.append({
                        "Product ID": pid,
                        "Brand": p.get("brand"),
                        "Product Name": p.get("product_name"),
                        "Price": f"₹{int(p.get('price_inr', 0))}",
                        "Cosine Distance": dist_str,
                        "Semantic Sim": f"{p.get('semantic_similarity', 0.0):.4f}",
                        "Hybrid Match": f"{p.get('match_percentage', 0)}%"
                    })
                st.dataframe(diag_rows, use_container_width=True)


if __name__ == "__main__":
    run_app()
