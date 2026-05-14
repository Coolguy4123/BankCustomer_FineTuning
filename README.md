# Customer Complaint Response Fine-Tuning (Applied Project)

## Project Overview

This is an **applied machine learning project** focused on building a practical system for generating professional banking customer support responses, showcasing the difference between a base model vs fine-tuned model.

The CFPB banking complaint dataset from Kaggle contains complaint narratives but does not include corresponding support responses. Therefore, synthetic responses were generated using a local Llama 3.1 8B model. These responses were then used to fine-tune a smaller model (Qwen2.5-1.5B-Instruct) using QLoRA and LoRA through the Unsloth library.

The goal of this project is to design and implement a complete ML pipeline that:
- generates training data
- fine-tunes a model efficiently
- deploys a working application

---

## Pipeline Summary

1. Load CFPB complaint dataset and extract `complaints.csv` from Kaggle: https://www.kaggle.com/datasets/adhamelkomy/bank-customer-complaint-analysis/data
2. Generate synthetic responses using Llama 3.1 8B (local Ollama)  
3. Filter low-quality responses  
4. Fine-tune Qwen2.5-1.5B using QLoRA (Unsloth)  
5. Compare base vs fine-tuned outputs  
6. Deploy with Python Streamlit  

---

## File Structure

```bash
data/
└── Complaint_SyntheticResponse_1k.csv   # Filtered dataset after running KAGGLE_BankingComplaint-PEFT.ipynb locally (~600 clean samples)

notebooks/
├── LOCAL_Synthesis_notebook.ipynb       # Synthetic data generation (Ollama)
└── KAGGLE_BankingComplaint-PEFT.ipynb   # Fine-tuning (Qwen + Unsloth + QLoRA)

app.py                                  # Streamlit demo (base vs fine-tuned)
.env # For HuggingFace Token
.env.example # Example for env
README.md
requirements.txt
```
---

## How to Run
### 1. Set up HuggingFace Token in .env
Input your HuggingFace token, example is provided in `.env.example` and import your HF Token for the StremlitDemo to work

---

### 2. Run Notebooks 
#### a. [Optional] Synthetic Data Generation (Local)

** The synthetic dataset was already generated with 1000 samples. If needed to run with different configurations, continue reading the setup for this notebook and run it first. 

Run locally using Ollama:
```bash
notebooks/LOCAL_Synthesis_notebook.ipynb
```

Requirements:
- Ollama installed  
- Model: `llama3.1:8b`  


#### b. Fine-Tuning (Recommended running Kaggle)

Run this notebook on **Kaggle with GPU** since file paths are in kaggle format and Kaggle offers GPU acceleration:
```bash
notebooks/KAGGLE_BankingComplaint-PEFT.ipynb
```

Steps:
1. Upload dataset (`Complaint_SyntheticResponse_1k.csv`) to Kaggle  
2. Update dataset path in notebook  
3. Run all cells  

---

### 3. Demo (Streamlit)

Run locally:

```bash
streamlit run app.py
# or
python -m streamlit run app.py # Windows
# or
python3 -m streamlit run app.py # Mac
```