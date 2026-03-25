import streamlit as st
import requests
import PyPDF2

# 1. Configure the page settings
st.set_page_config(page_title="HalluciGuard", page_icon="🛡️", layout="centered")

# --- CUSTOM Premium CSS ---
st.markdown("""
<style>
/* Global Font Import */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Glassmorphism for text areas */
.stTextArea textarea {
    background-color: rgba(22, 27, 34, 0.6) !important;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    padding: 12px !important;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.3) !important;
}

/* Beautiful Button Gradient */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF 0%, #3B33D4 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.4) !important;
}

/* Glassmorphism Metric Cards */
div[data-testid="stMetric"] {
    background-color: rgba(22, 27, 34, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
}

/* Fancy Header Gradient */
h1 {
    background: -webkit-linear-gradient(45deg, #A78BFA, #6C63FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    margin-bottom: 0rem !important;
    padding-bottom: 1rem;
}

/* File Uploader tweaking */
div[data-testid="stFileUploader"] {
    background-color: rgba(22, 27, 34, 0.4);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px dashed rgba(255, 255, 255, 0.15);
    transition: all 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #6C63FF;
    background-color: rgba(108, 99, 255, 0.05);
}

/* Subheaders */
h3 {
    color: #e6edf3 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ HalluciGuard")
st.write("Research-grade LLM Hallucination Detection using NLI and Cross-Encoders.")

# 2. Define the Backend URL
API_URL = "http://127.0.0.1:8000/api/v1/score"

# 3. Create the Input Forms
st.subheader("Test an LLM Output")

# Initialize session state for context text and uploaded file tracking
if "context_text" not in st.session_state:
    st.session_state.context_text = ""
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

uploaded_file = st.file_uploader("Upload a document to use as Ground Truth (Optional)", type=["pdf"])

if uploaded_file is not None:
    # Only process the file if it's a new upload or newly selected
    if st.session_state.uploaded_filename != uploaded_file.name:
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            extracted_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            
            if not extracted_text.strip():
                st.warning("Could not extract text. Please ensure the PDF is text-searchable and not a scanned image.")
            else:
                st.session_state.context_text = extracted_text.strip()
                st.session_state.uploaded_filename = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

context = st.text_area(
    "Source Context (The factual ground truth):", 
    key="context_text",
    height=150, 
    placeholder="e.g., The Q3 financial report states that the company's revenue grew by 15%, reaching $50 million."
)

llm_output = st.text_area(
    "LLM Output (The generated response to evaluate):", 
    height=100, 
    placeholder="e.g., The company had a great Q3, bringing in $60 million in revenue."
)

# 4. The Action Button
if st.button("Detect Hallucination", type="primary", use_container_width=True):
    if not context or not llm_output:
        st.warning("⚠️ Please provide both a context and an LLM output to test.")
    else:
        # Show a premium animated status indicator while the API processes
        with st.status("🔍 Extracting text and structuring context...", expanded=False) as status:
            st.write("⏳ Creating overlapping sequence chunks (evading 512-token limit)...")
            st.write("🧠 Evaluating chunk semantics against DeBERTa-v3 cross-encoder...")
            
            try:
                # Send data to our FastAPI backend
                response = requests.post(
                    API_URL, 
                    json={"context": context, "llm_output": llm_output}
                )
                response.raise_for_status() # Throw an error if the API crashes
                
                status.update(label="✅ Analysis Complete & Scores Aggregated!", state="complete", expanded=True)
                
                data = response.json()
                results = data.get("results", {})

                # 5. Display the Results
                st.markdown("---")
                
                # Big visual alert
                if results.get("is_hallucination"):
                    st.error("🚨 **HALLUCINATION DETECTED (Contradiction > 60%)**")
                else:
                    st.success("✅ **FACTUALLY CONSISTENT**")
                    
                # Display exact percentages cleanly
                st.subheader("NLI Confidence Scores")
                
                contradiction_score = results.get('contradiction_score', 0)
                entailment_score = results.get('entailment_score', 0)
                neutral_score = results.get('neutral_score', 0)
                
                # Dynamic Plain-English Summary
                if contradiction_score > 15:
                    st.warning("🚨 Warning: The AI is explicitly contradicting the source material.")
                elif neutral_score > 50:
                    st.warning("⚠️ Note: The AI is going off-script. It is bringing in outside knowledge not found in your source context.")
                elif entailment_score > 50 and contradiction_score < 5:
                    st.success("✅ Excellent: The AI is sticking strictly to the facts provided.")
                    
                # Generate dynamic contextual captions based on percentages
                if contradiction_score >= 60:
                    c_text = "- Meaning: The AI is heavily contradicting the source facts. Critical hallucination risk."
                elif contradiction_score > 10:
                    c_text = "- Meaning: Some distinct parts of the output directly oppose the source material."
                else:
                    c_text = "- Meaning: The AI is largely staying true to the facts with zero serious contradictions."
                
                if entailment_score >= 80:
                    e_text = "- Meaning: Excellent! The AI is strictly summarizing the facts given to it."
                elif entailment_score > 40:
                    e_text = "- Meaning: The AI includes a fair mix of factual summaries alongside some conversational filler."
                else:
                    e_text = "- Meaning: Very little of the output is strictly supported by the source text."
                    
                if neutral_score >= 60:
                    n_text = "- Meaning: The AI is heavily going off-script and bringing in outside knowledge/questions."
                elif neutral_score > 10:
                    n_text = "- Meaning: The AI is adding a moderate amount of conversational filler or harmless additions."
                else:
                    n_text = "- Meaning: The AI is strictly sticking to the provided context with almost no unverified additions."
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Contradiction", f"{contradiction_score}%")
                    st.caption(c_text)
                with col2:
                    st.metric("Entailment", f"{entailment_score}%")
                    st.caption(e_text)
                with col3:
                    st.metric("Neutral", f"{neutral_score}%")
                    st.caption(n_text)
                


            except requests.exceptions.ConnectionError:
                st.error("🔌 Failed to connect to the backend. Is your FastAPI server running on port 8000?")
            except Exception as e:
                st.error(f"An error occurred: {e}")