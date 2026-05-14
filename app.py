import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---- Configurations ----
BASE_MODEL_NAME    = "Qwen/Qwen2.5-1.5B-Instruct"
HF_REPO            = "Coolguy41234/qwen-bank-support"
MAX_NEW_TOKENS     = 200
TEMPERATURE        = 0.25
TOP_P              = 0.90
REPETITION_PENALTY = 1.15

SYSTEM_PROMPT = (
    "You are a senior banking customer support specialist. "
    "Write a professional, empathetic, and specific response to the customer complaint. "
    "Clearly acknowledge the issue, use product-relevant language, and explain likely next steps. "
    "Do not claim the bank has completed an investigation. "
    "Do not promise compensation or legal outcomes."
)

EXAMPLE_COMPLAINTS = [
    "I was charged an overdraft fee even though my linked savings account had enough money to cover the transaction. I have overdraft protection set up and this has never happened before.",
    "My mortgage payment was reported as late to the credit bureaus but I have proof from your online portal that it was submitted on time. This is damaging my credit score.",
    "I disputed a charge on my credit card three weeks ago and I still have not received any update. The charge was for a service I never received.",
    "I applied for a personal loan and was denied without any explanation. I have a good credit history and steady income. I need to understand why.",
]


# ----- Device detection -----
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dtype():
    if torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.float16
    return torch.float32


# ---- Model Loading and Cache for faster speeds ----
@st.cache_resource(show_spinner=False)
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

@st.cache_resource(show_spinner=False)
def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=get_dtype(),
        device_map={"": get_device()},
    )
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_finetuned_model():
    model = AutoModelForCausalLM.from_pretrained(
        HF_REPO,
        torch_dtype=get_dtype(),
        device_map={"": get_device()},
    )
    model.eval()
    return model


# ----- Inference -----
def build_prompt(tokenizer, complaint: str, product: str = "") -> str:
    if product:
        user_content = f"Product category: {product}\n\nCustomer complaint:\n{complaint}"
    else:
        user_content = f"Customer complaint:\n{complaint}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

@torch.no_grad()
def generate(model, tokenizer, complaint: str, product: str = "") -> str:
    prompt = build_prompt(tokenizer, complaint, product)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, im_end_id],
        use_cache=True,
    )

    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ----- Streamlit UI -----
st.set_page_config(
    page_title="CS4210 — Base vs Fine-tuned Qwen",
    page_icon="🏦",
    layout="wide",
)

st.markdown("""
<style>
    .response-card {
        padding: 1rem;
        border-radius: 8px;
        white-space: pre-wrap;
        color: #1a1a1a !important;
        font-size: 0.95rem;
        line-height: 1.6;
        min-height: 80px;
    }
    .response-base {
        background: #f8f9fa;
        border-left: 4px solid #adb5bd;
    }
    .response-ft {
        background: #f0fff4;
        border-left: 4px solid #40c057;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Banking Complaint Response — Base vs Fine-tuned")
st.markdown(
    "Enter a customer complaint and compare how the **base model** responds "
    "versus the **fine-tuned model** trained on CFPB banking support responses."
)

# - Sidebar -
with st.sidebar:
    st.header("Model info")
    st.markdown("**Base model:**")
    st.code(BASE_MODEL_NAME, language=None)
    st.markdown("**Fine-tuned:**")
    st.code(HF_REPO, language=None)
    st.markdown("---")
    st.markdown(f"**Max new tokens:** `{MAX_NEW_TOKENS}`")
    st.markdown(f"**Temperature:** `{TEMPERATURE}`")
    st.markdown(f"**Top-p:** `{TOP_P}`")
    st.markdown(f"**Rep. penalty:** `{REPETITION_PENALTY}`")
    st.markdown("---")
    device = get_device()
    device_label = {"cuda": "GPU (CUDA)", "mps": "GPU (Apple MPS)", "cpu": "CPU"}
    st.markdown(f"**Hardware:** {device_label.get(device, device.upper())}")
    if device == "cpu":
        st.warning(
            "Running on CPU. Generation will take 60-90 seconds per model. "
            "Click Generate once before your demo to warm up the cache."
        )

st.subheader("Customer complaint")

example_choice = st.selectbox(
    "Load an example or type your own:",
    ["(type your own)"] + EXAMPLE_COMPLAINTS,
    index=0,
)

default_text = "" if example_choice == "(type your own)" else example_choice
complaint_text = st.text_area(
    "Complaint narrative",
    value=default_text,
    height=140,
    placeholder="Describe the customer complaint here...",
)

product_text = st.text_input(
    "Product category (optional)",
    placeholder="e.g. Mortgage, Credit card, Checking account",
)

generate_button = st.button(
    "Generate responses", type="primary", use_container_width=True
)

# Output columns
col_base, col_ft = st.columns(2)

with col_base:
    st.markdown("### Base model")
    st.caption("Qwen2.5-1.5B-Instruct — no fine-tuning")
    base_output = st.empty()

with col_ft:
    st.markdown("### Fine-tuned model")
    st.caption("QLoRA · rsLoRA r=16 · CFPB synthetic targets")
    ft_output = st.empty()

if generate_button:
    if not complaint_text.strip():
        st.warning("Please enter a complaint before generating.")
        st.stop()

    tokenizer = load_tokenizer()

    with col_base:
        with st.spinner("Base model generating..."):
            base_model = load_base_model()
            base_response = generate(base_model, tokenizer, complaint_text, product_text)
        base_output.markdown(
            '<div class="response-card response-base">' + base_response + '</div>',
            unsafe_allow_html=True,
        )

    with col_ft:
        with st.spinner("Fine-tuned model generating..."):
            ft_model = load_finetuned_model()
            ft_response = generate(ft_model, tokenizer, complaint_text, product_text)
        ft_output.markdown(
            '<div class="response-card response-ft">' + ft_response + '</div>',
            unsafe_allow_html=True,
        )

    # Metrics row
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base word count",       len(base_response.split()))
    m2.metric("Fine-tuned word count", len(ft_response.split()))
    m3.metric(
        "Base ends cleanly",
        "Yes" if base_response.strip() and base_response.strip()[-1] in ".!?" else "No",
    )
    m4.metric(
        "FT ends cleanly",
        "Yes" if ft_response.strip() and ft_response.strip()[-1] in ".!?" else "No",
    )