import time
import streamlit as st
import pandas as pd
import os
import sys
import io
import glob
import subprocess

# Add current directory to path so imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Load existing environment variables
from dotenv import load_dotenv
load_dotenv()

# Pricing per million tokens
MODEL_PRICING = {
    # Gemini
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Anthropic
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    # Groq
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b": {"input": 0.27, "output": 0.27},
}

# ---------------------------------------------------------
# Monkey-patch Retrieval
# ---------------------------------------------------------
from src.retrieval.hybrid_search import HybridSearch

def patched_hybrid_search(self, query: str, company: str | None = None, filters: dict | None = None):
    search_mode = st.session_state.get("retrieval_mode", "Hybrid search (default)")
    current_top_k = st.session_state.get("final_top_k", 5)
    
    vector_filters = self._build_filters(company=company, filters=filters)
    
    if search_mode == "Only Vector search":
        return self.vector.search(query=query, top_k=current_top_k, filters=vector_filters)
    elif search_mode == "Only BM25 search":
        return self.bm25.search(query=query, top_k=current_top_k, company=company)
    else:
        # Hybrid
        bm25_res = self.bm25.search(query=query, top_k=current_top_k * 2, company=company)
        vector_res = self.vector.search(query=query, top_k=current_top_k * 2, filters=vector_filters)
        fused = self.rrf.fuse(bm25=bm25_res, vector=vector_res)
        return fused[:current_top_k]

HybridSearch.search = patched_hybrid_search

# ---------------------------------------------------------
# Monkey-patch LLMClient to collect metrics
# ---------------------------------------------------------
import src.ai.client as client
original_generate = client.LLMClient.generate

def patched_generate(self, messages, response_schema):
    start_time = time.time()
    
    # Input tokens estimation
    input_text = ""
    for msg in messages:
        if hasattr(msg, "content"):
            input_text += str(msg.content)
        else:
            input_text += str(msg)
    input_tokens = max(1, int(len(input_text) / 4))
    
    # Execute original LLM call
    result = original_generate(self, messages, response_schema)
    
    elapsed = time.time() - start_time
    output_text = str(result)
    output_tokens = max(1, int(len(output_text) / 4))
    total_tokens = input_tokens + output_tokens
    
    # Cost calculation
    model_name = st.session_state.get("selected_model", "gemini-2.5-flash-lite")
    pricing = MODEL_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    
    if "current_stats" not in st.session_state:
        st.session_state.current_stats = []
        
    st.session_state.current_stats.append({
        "elapsed": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost": cost
    })
    
    return result

client.LLMClient.generate = patched_generate

# Now import the support agent and state components
from src.graph.support_agent import SupportAgent
from src.ai.models import SupportTicket
import src.graph.nodes as nodes
from src.ai.client import LLMClient
import src.config as src_config

# ---------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------
st.set_page_config(page_title="SupportSphere AI", page_icon="🤖", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .main .block-container { padding-top: 2rem; }
    .stMetric { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 12px; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Options (Exactly as before)
# ---------------------------------------------------------
st.sidebar.header("🧠 LLM & API Key Settings")

provider = st.sidebar.selectbox(
    "Provider",
    options=["Google", "OpenAI", "Anthropic", "Groq"]
)

# Model selection based on provider
model_options = {
    "Google": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Anthropic": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
    "Groq": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
}

selected_model = st.sidebar.selectbox(
    "Model",
    options=model_options.get(provider, ["gemini-2.5-flash-lite"]),
    key="selected_model"
)

# Map provider to corresponding API key environment variable name
provider_key_map = {
    "Google": "GOOGLE_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Groq": "GROQ_API_KEY"
}

# Initialize session state keys to empty strings by default
for key in ["GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"]:
    if key not in st.session_state:
        st.session_state[key] = os.getenv(key, "")

# Conditionally show the text input for API Key
google_key = st.session_state["GOOGLE_API_KEY"]
openai_key = st.session_state["OPENAI_API_KEY"]
anthropic_key = st.session_state["ANTHROPIC_API_KEY"]
groq_key = st.session_state["GROQ_API_KEY"]

key_missing = False
if provider in provider_key_map:
    target_key = provider_key_map[provider]
    entered_key_val = st.sidebar.text_input(
        f"Enter {target_key}",
        value=st.session_state[target_key],
        type="password",
        help=f"Provide your {target_key} credential."
    )
    if entered_key_val != st.session_state[target_key]:
        st.session_state[target_key] = entered_key_val
        
    # Re-assign keys with the updated values
    google_key = st.session_state["GOOGLE_API_KEY"]
    openai_key = st.session_state["OPENAI_API_KEY"]
    anthropic_key = st.session_state["ANTHROPIC_API_KEY"]
    groq_key = st.session_state["GROQ_API_KEY"]

    key_missing = not st.session_state[target_key].strip()
    if key_missing:
        st.sidebar.error(f"⚠️ {target_key} is required to run the models.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Retrieval Settings")

# Slider to adjust FINAL_TOP_K with default as 5
final_top_k = st.sidebar.slider(
    "FINAL_TOP_K",
    min_value=1,
    max_value=20,
    value=5,
    key="final_top_k",
    help="Number of final retrieved document chunks to include in LLM context."
)

retrieval_mode = st.sidebar.selectbox(
    "Search Mode",
    options=["Hybrid search (default)", "Only Vector search", "Only BM25 search"],
    key="retrieval_mode"
)

# Apply settings dynamically to internal configuration
src_config.LLM_PROVIDER = provider.lower()
src_config.LLM_MODEL = selected_model
src_config.GOOGLE_API_KEY = google_key
src_config.OPENAI_API_KEY = openai_key
src_config.ANTHROPIC_API_KEY = anthropic_key
src_config.GROQ_API_KEY = groq_key
src_config.FINAL_TOP_K = final_top_k

# Update environment variables
os.environ["LLM_PROVIDER"] = provider.lower()
os.environ["LLM_MODEL"] = selected_model
os.environ["GOOGLE_API_KEY"] = google_key
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["ANTHROPIC_API_KEY"] = anthropic_key
os.environ["GROQ_API_KEY"] = groq_key

# ---------------------------------------------------------
# Helper: Run Ticket Processing
# ---------------------------------------------------------
def run_triage(issue, subject, company):
    nodes.llm = LLMClient()
    agent = SupportAgent()
    ticket = SupportTicket(
        issue=issue,
        subject=subject or "",
        company=company or "None"
    )
    
    st.session_state.current_stats = []
    
    start_time = time.time()
    result = agent.invoke(ticket)
    total_time = time.time() - start_time
    
    stats = st.session_state.get("current_stats", [{"elapsed": total_time, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}])[0]
    stats["elapsed"] = total_time
    
    return result, stats

# ---------------------------------------------------------
# Main App Layout with Tabs
# ---------------------------------------------------------
st.title("🤖 SupportSphere AI Triage Hub")
st.write("An enterprise support triage assistant for HackerRank, Claude, and Visa tickets.")

tab_home, tab1, tab2, tab_eval = st.tabs([
    "🏠 Home", 
    "🎫 Single Ticket Triage", 
    "📁 Batch File Triage", 
    "📊 Evaluation"
])

# =========================================================
# Tab: Home
# =========================================================
with tab_home:
    st.header("System Overview")
    st.write("Welcome to the SupportSphere AI Triage Hub.")
    st.markdown("""
    SupportSphere AI utilizes advanced Retrieval-Augmented Generation (RAG) and structured outputs to analyze customer support tickets, identify product areas, check consistency, and write grounded customer-facing drafts.
    
    ### Supported Knowledge Bases
    - **HackerRank**: Tests, Assessments, Library, Integrations
    - **Claude**: Features, Out-of-Scope, Privacy, Account Management
    - **Visa**: Travel Support, General Card Support
    
    ### System Capabilities
    1. **Dynamic Hybrid Retrieval**: Combines BM25 lexical keyword matching and vector search.
    2. **LangGraph Pipeline**: Runs input checks, retrieves documents, validates retrieval, generates response, and runs output guardrail checks.
    3. **Observability**: Complete validation warnings logging, performance tracking, and latency profiling.
    """)
    
    # Try to load high-level metrics for the current model
    current_csv = f"evaluation/reports/evaluation_results_{provider.lower()}_{selected_model.lower().replace('/', '_')}.csv"
    if os.path.exists(current_csv):
        try:
            df = pd.read_csv(current_csv)
            st.markdown("### 📈 Recent Evaluation Summary")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tickets Evaluated", len(df))
            col2.metric("Request Type Accuracy", f"{df['request_type_accuracy'].mean():.0%}")
            col3.metric("Product Area Accuracy", f"{df['product_area_accuracy'].mean():.0%}")
            col4.metric("Status Accuracy", f"{df['status_accuracy'].mean():.0%}")
        except Exception:
            pass

# =========================================================
# Tab: Single Ticket Triage
# =========================================================
with tab1:
    st.header("Single Ticket Input")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        issue_input = st.text_area("Ticket Issue / Body", placeholder="Describe the user's issue...", height=200)
    with col2:
        subject_input = st.text_input("Subject (Optional)", placeholder="e.g. Password reset fail")
        company_input = st.selectbox("Company", options=["HackerRank", "Claude", "Visa", "None"])
    if key_missing:
        st.warning(f"⚠️ Please enter your {provider_key_map[provider]} in the sidebar to enable ticket processing.")
        
    if st.button("🚀 Process Ticket", type="primary", disabled=key_missing):
        if not issue_input.strip():
            st.error("Please enter a ticket issue.")
        else:
            with st.spinner("Analyzing and retrieving context..."):
                result, stats = run_triage(issue_input, subject_input, company_input)
                
            # Metrics
            st.markdown("### 📊 Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Inference Time", f"{stats['elapsed']:.2f}s")
            m_col2.metric("Input / Output Tokens", f"{stats['input_tokens']} / {stats['output_tokens']}")
            m_col3.metric("Total Tokens", f"{stats['total_tokens']}")
            m_col4.metric("Estimated Cost", f"${stats['cost']:.5f}")
            
            # Outputs
            st.markdown("---")
            st.markdown("### 🏷️ Triage Fields")
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.markdown(f"**Request Type:** `{result.response.request_type}`")
            f_col2.markdown(f"**Product Area:** `{result.response.product_area}`")
            f_col3.markdown(f"**Status:** `{result.response.status}`")
            
            st.markdown("### ✍️ Generated Response")
            st.info(result.response.response)
            
            st.markdown("### 💡 Justification")
            st.code(result.response.justification)

            # Reference Chunks
            if result.sources:
                st.markdown("### 🔗 Reference Sources")
                for src in result.sources:
                    st.markdown(f"- [{src.title}]({src.url}) ({src.company})")

# =========================================================
# Tab: Batch File Triage
# =========================================================
with tab2:
    st.header("Batch File Input")
    st.write("Upload a CSV file containing `Issue`, `Subject`, and `Company` columns.")
    
    uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Preview:")
            st.dataframe(df.head(5))
            
            # Check required columns
            required_cols = ["Issue", "Subject", "Company"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}")
            else:
                if key_missing:
                    st.warning(f"⚠️ Please enter your {provider_key_map[provider]} in the sidebar to enable batch processing.")
                if st.button("🏁 Run Batch Process", type="primary", disabled=key_missing):
                    results_list = []
                    total_elapsed = 0.0
                    total_input = 0
                    total_output = 0
                    total_cost = 0.0
                    
                    progress_bar = st.progress(0)
                    total_rows = len(df)
                    
                    for idx, row in df.iterrows():
                        issue = str(row["Issue"])
                        subject = str(row["Subject"]) if pd.notna(row["Subject"]) else ""
                        company = str(row["Company"]) if pd.notna(row["Company"]) else "None"
                        
                        # Process
                        result, stats = run_triage(issue, subject, company)
                        
                        # Accumulate stats
                        total_elapsed += stats["elapsed"]
                        total_input += stats["input_tokens"]
                        total_output += stats["output_tokens"]
                        total_cost += stats["cost"]
                        
                        # Add results
                        results_list.append({
                            "Issue": issue,
                            "Subject": subject,
                            "Company": company,
                            "Response": result.response.response,
                            "Product Area": result.response.product_area,
                            "Status": result.response.status,
                            "Request Type": result.response.request_type
                        })
                        
                        progress_bar.progress((idx + 1) / total_rows)
                    
                    # Display cumulative metrics
                    st.markdown("### 📊 Batch Run Metrics")
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Total Execution Time", f"{total_elapsed:.2f}s")
                    m_col2.metric("Total Input / Output Tokens", f"{total_input} / {total_output}")
                    m_col3.metric("Total Tokens Used", f"{total_input + total_output}")
                    m_col4.metric("Total Run Cost", f"${total_cost:.5f}")
                    
                    # Output DataFrame
                    out_df = pd.DataFrame(results_list)
                    st.markdown("---")
                    st.markdown("### 🎉 Processed Results")
                    st.dataframe(out_df)
                    
                    # Prepare Downloadable CSV
                    csv_buffer = io.StringIO()
                    out_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name="support_tickets_output.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error reading or processing CSV: {str(e)}")

# =========================================================
# Tab: Evaluation
# =========================================================
with tab_eval:
    st.header("📊 Model Evaluation & Comparison")
    st.write(f"Currently active model configuration: **{provider} ({selected_model})**")
    
    # Button to trigger runner
    if key_missing:
        st.warning(f"⚠️ Please set API credentials in the sidebar to enable running the evaluation pipeline.")
    
    if st.button("🔄 Run Evaluation Pipeline", disabled=key_missing):
        with st.spinner(f"Running evaluation runner on {provider} ({selected_model})..."):
            try:
                # Build environment
                env = os.environ.copy()
                env["LLM_PROVIDER"] = provider.lower()
                env["LLM_MODEL"] = selected_model
                env["GOOGLE_API_KEY"] = google_key
                env["OPENAI_API_KEY"] = openai_key
                env["ANTHROPIC_API_KEY"] = anthropic_key
                env["GROQ_API_KEY"] = groq_key
                env["FINAL_TOP_K"] = str(final_top_k)
                
                res = subprocess.run([sys.executable, "-m", "evaluation.runner"], capture_output=True, text=True, env=env, check=True)
                st.success("Evaluation completed successfully!")
                st.text(res.stdout)
            except Exception as e:
                st.error(f"Failed to run evaluation pipeline: {e}")
                
    current_csv = f"evaluation/reports/evaluation_results_{provider.lower()}_{selected_model.lower().replace('/', '_')}.csv"
    
    if os.path.exists(current_csv):
        try:
            df = pd.read_csv(current_csv)
            
            # Global Metrics Cards
            st.markdown("### 🏆 Active Configuration Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Tickets", len(df))
            col2.metric("Request Type Accuracy", f"{df['request_type_accuracy'].mean():.0%}")
            col3.metric("Product Area Accuracy", f"{df['product_area_accuracy'].mean():.0%}")
            col4.metric("Status Accuracy", f"{df['status_accuracy'].mean():.0%}")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Avg Correctness", f"{df['answer_correctness'].mean():.2f}")
            col6.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.2f}")
            
            latency_avg = df["latency"].mean() if "latency" in df.columns else 0.0
            col7.metric("Avg Latency", f"{latency_avg:.2f}s")
            col8.metric("Avg Chunks Retrieved", f"{df['num_chunks'].mean():.1f}")
            
            # Build pass metric
            df['Pass'] = df.apply(
                lambda row: "✅" if (
                    row['request_type_accuracy'] == 1 and
                    row['product_area_accuracy'] == 1 and
                    row['status_accuracy'] == 1 and
                    row['answer_correctness'] >= 0.8 and
                    row['faithfulness'] >= 0.8
                ) else "⚠️", axis=1
            )
            
            # Results Table
            st.markdown("### 🎫 Ticket Results Table")
            table_df = pd.DataFrame({
                "Ticket": [f"#{i+1}" for i in range(len(df))],
                "Request Type": ["✅" if r == 1 else "❌" for r in df["request_type_accuracy"]],
                "Product Area": ["✅" if p == 1 else "❌" for p in df["product_area_accuracy"]],
                "Status": ["✅" if s == 1 else "❌" for s in df["status_accuracy"]],
                "Correctness": df["answer_correctness"].round(2),
                "Faithfulness": df["faithfulness"].round(2),
                "Pass": df["Pass"]
            })
            st.dataframe(table_df, use_container_width=True)
            
            # Show Detailed Row Inspection
            st.markdown("### 🔍 Detailed Ticket Inspection")
            selected_ticket_idx = st.selectbox("Select Ticket to Inspect", options=range(1, len(df)+1))
            row_data = df.iloc[selected_ticket_idx - 1]
            
            st.markdown(f"**Ticket #{selected_ticket_idx} Details:**")
            st.write(f"- **Company:** `{row_data.get('company')}`")
            st.write(f"- **Subject:** `{row_data.get('subject')}`")
            st.markdown(f"**Issue / Inquiry:**")
            st.info(row_data.get("issue"))
            
            st.markdown(f"**LLM-as-a-Judge Reasoning:**")
            st.code(row_data.get("judge_reasoning", "No explanation available."))
            
        except Exception as e:
            st.error(f"Error loading evaluation results CSV: {e}")
    else:
        st.warning(f"⚠️ No evaluation results CSV found for the active config **{provider} ({selected_model})**. Click the button above to run the pipeline.")
        
    # ---------------------------------------------------------
    # Comparison Section
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Cross-Model Comparison")
    
    # Scan for files
    all_reports = glob.glob("evaluation/reports/evaluation_results_*.csv")
    if all_reports:
        comparison_rows = []
        for path in all_reports:
            filename = os.path.basename(path)
            parts = filename.replace("evaluation_results_", "").replace(".csv", "").split("_")
            if len(parts) >= 2:
                model_prov = parts[0].upper()
                model_name = "_".join(parts[1:])
            else:
                model_prov = "UNKNOWN"
                model_name = filename
                
            try:
                comp_df = pd.read_csv(path)
                latency_val = comp_df["latency"].mean() if "latency" in comp_df.columns else 0.0
                comparison_rows.append({
                    "Provider": model_prov,
                    "Model": model_name,
                    "Req Type Acc": f"{comp_df['request_type_accuracy'].mean():.0%}",
                    "Prod Area Acc": f"{comp_df['product_area_accuracy'].mean():.0%}",
                    "Status Acc": f"{comp_df['status_accuracy'].mean():.0%}",
                    "Avg Correctness": round(comp_df['answer_correctness'].mean(), 2),
                    "Avg Faithfulness": round(comp_df['faithfulness'].mean(), 2),
                    "Avg Latency": f"{latency_val:.2f}s",
                    "Avg Chunks": round(comp_df['num_chunks'].mean(), 1)
                })
            except Exception:
                pass
                
        if comparison_rows:
            comp_table = pd.DataFrame(comparison_rows)
            st.dataframe(comp_table, use_container_width=True)
        else:
            st.info("No comparative runs available yet.")
    else:
        st.info("No reports found to build a comparison list.")
