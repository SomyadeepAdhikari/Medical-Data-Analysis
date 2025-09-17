import os
import json
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import importlib

# Diagnostics: verify torch availability early
if importlib.util.find_spec("torch") is None:
    print("[WARNING] torch not installed. Install with: pip install torch --upgrade")
else:
    import torch  # noqa: E402
    print(f"[INFO] torch version: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")

# Load environment variables
load_dotenv()

# Define the path to the Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

"""
RAG Interactive QA Script (Free Stack)

Changes:
- Removed Google Generative AI usage.
- Uses HuggingFace embeddings (default: sentence-transformers/all-MiniLM-L6-v2).
- Uses a HuggingFace causal LM for generation (default: mistralai/Mistral-7B-Instruct-v0.3; change via env HUGGINGFACE_LLM_MODEL).
- Adds optional Sarvam AI translation for multilingual Q&A if SARVAM_API_KEY is present.

Environment Variables (update your .env):
    HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    HUGGINGFACE_LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
    HUGGINGFACE_DEVICE=cpu  # or cuda
    LLM_TEMPERATURE=0.2
    SARVAM_API_KEY=your_key_here
    SARVAM_BASE_URL=https://api.sarvam.ai
"""

# Initialize embeddings and vector store (HuggingFace)
embedding_model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

hf_model_name = os.getenv("HUGGINGFACE_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
temperature = float(os.getenv("LLM_TEMPERATURE", 0.2))
skip_model = os.getenv("SKIP_LLM", "false").lower() in {"1", "true", "yes"}

generation_pipeline = None
parser = StrOutputParser()

def load_generation_pipeline():
    global generation_pipeline
    if generation_pipeline is not None:
        return generation_pipeline
    if skip_model:
        print("[INFO] SKIP_LLM=true -> Skipping model load. Retrieval only mode.")
        return None
    print(f"Loading HuggingFace model: {hf_model_name} on {device}")
    if device != "cpu":
        print("[INFO] If you encounter CUDA OOM, set HUGGINGFACE_DEVICE=cpu or choose a smaller model.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        model_ref = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            device_map="auto" if device != "cpu" else None,
        )
        generation_pipeline = pipeline(
            "text-generation",
            model=model_ref,
            tokenizer=tokenizer,
            device=0 if device != "cpu" else -1,
            max_new_tokens=512,
            temperature=temperature,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model {hf_model_name}: {e}\nFalling back to 'distilgpt2'.")
        try:
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            model_ref = AutoModelForCausalLM.from_pretrained("distilgpt2")
            generation_pipeline = pipeline(
                "text-generation",
                model=model_ref,
                tokenizer=tokenizer,
                device=0 if device != "cpu" else -1,
                max_new_tokens=256,
                temperature=temperature,
            )
        except Exception as e2:
            print(f"[FATAL] Could not load fallback model: {e2}. You can set HUGGINGFACE_LLM_MODEL=sshleifer/tiny-gpt2 for a very small test model.")
            generation_pipeline = None
    return generation_pipeline

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_BASE = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")

def sarvam_translate(text: str, target_lang: str = "en") -> str:
    if not SARVAM_API_KEY:
        return text
    try:
        url = f"{SARVAM_BASE}/v1/translate"
        payload = {"text": text, "target_language": target_lang}
        headers = {"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("translated_text", text)
        return text
    except Exception:
        return text

def detect_language_simple(text: str) -> str:
    ascii_ratio = sum(c.isascii() for c in text) / max(1, len(text))
    if ascii_ratio > 0.95:
        return "en"
    return "unknown"

# Set up initial message context
messages = [SystemMessage(content="You are a helpful assistant that analyzes health report data.")]

# Start interactive loop
while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Language handling
    original_lang = detect_language_simple(query)
    working_query = query if original_lang == "en" else sarvam_translate(query, target_lang="en")

    # Retrieve relevant documents
    relevant_docs = retriever.invoke(working_query)
    docs_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build the prompt
    combined_prompt = (
        f"Question: {working_query}\n\n"
        f"Relevant Documents:\n{docs_text}\n\n"
        "Provide a concise factual answer based ONLY on the documents above. If the answer is not present, reply exactly: I'm not sure."
    )

    messages.append(HumanMessage(content=combined_prompt))

    pipe = load_generation_pipeline()
    if pipe is None:
        print("[WARN] No generation pipeline available; showing retrieved context only.")
        print("\n--- Retrieved Context ---\n")
        print(docs_text[:1500])
        continue
    try:
        raw_output = pipe(
            combined_prompt,
            do_sample=False,
            num_return_sequences=1,
        )[0]["generated_text"]
        response_text = raw_output[len(combined_prompt):].strip() or raw_output.strip()
    except Exception as gen_err:
        response_text = f"Generation failed: {gen_err}"
    if len(response_text) > 4000:
        response_text = response_text[:4000]
    if original_lang != "en":
        response_text = sarvam_translate(response_text, target_lang=original_lang)
    messages.append(AIMessage(content=response_text))

    # Print the response
    print("\n--- LLM Response ---")
    print(response_text)


""" 

Find the number of patients with a specific condition.
Get the list of patients who have undergone a specific test(Blood Glucose).
Find the patients who have a certain symptom(fatigue).
Find the oldest and youngest patients in the dataset.
Retrieve the patients who have a specific medication.
Get the list of patients who have a specific gender and age range.
Get the list of patients who have no symptoms.

Which condition were more common in ages 20-30 according to the dataset? 
How effective was Insulin based on the cases provided?
Are there common symptoms that appear across multiple conditions?
What are the main findings about Coronary Artery Disease in the dataset?
"""

