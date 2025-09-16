import os
import json
import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables
load_dotenv()

# Define the path to the Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embedding_model_name = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

query = os.getenv("SAMPLE_QUERY", "What are the main findings about Coronary Artery Disease in the dataset?")

# Get relevant documents from the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
relevant_docs = retriever.invoke(query)

# Combine documents into a prompt
docs_text = "\n\n".join([doc.page_content for doc in relevant_docs])
combined_prompt = (
    f"You are a helpful assistant that analyzes health report data.\n"
    f"Question: {query}\n\n"
    f"Relevant Documents:\n{docs_text}\n\n"
    "Please answer the question based ONLY on the documents above. "
    "If the answer is not present in the documents, respond with: 'I'm not sure.'"
)

hf_model_name = os.getenv("HUGGINGFACE_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
temperature = float(os.getenv("LLM_TEMPERATURE", 0.2))
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
parser = StrOutputParser()

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

# Send message to LLM
orig_lang = detect_language_simple(query)
working_prompt = combined_prompt if orig_lang == "en" else sarvam_translate(combined_prompt, target_lang="en")

messages = [SystemMessage(content="You are a helpful assistant that analyzes health report data."), HumanMessage(content=working_prompt)]

raw_output = generation_pipeline(
    working_prompt,
    do_sample=False,
    num_return_sequences=1,
)[0]["generated_text"]
response_text = raw_output[len(working_prompt):].strip() or raw_output.strip()
if orig_lang != "en":
    response_text = sarvam_translate(response_text, target_lang=orig_lang)

print("\n--- LLM Response ---")
print(response_text)



