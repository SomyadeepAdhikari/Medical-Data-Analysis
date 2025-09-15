import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Define the path to the Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Define the query
query = "What are the main findings about Coronary Artery Disease in the dataset?"

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

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")  # or "gemini-2.0-pro" if available
parser = StrOutputParser()

# Send message to LLM
messages = [
    SystemMessage(content="You are a helpful assistant that analyzes health report data."),
    HumanMessage(content=combined_prompt)
]

# Invoke the model and parse the result
response = (model | parser).invoke(messages)

# Print the result
print("\n--- LLM Response ---")
print(response)



