import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv()

# Define the path to the Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Initialize retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the Gemini model and parser
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
parser = StrOutputParser()

# Set up initial message context
messages = [
    SystemMessage(content="You are a helpful assistant that analyzes health report data.")
]

# Start interactive loop
while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant documents
    relevant_docs = retriever.invoke(query)
    docs_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Build the prompt
    combined_prompt = (
        f"Question: {query}\n\n"
        f"Relevant Documents:\n{docs_text}\n\n"
        "Please answer the question based ONLY on the documents above. "
        "If the answer is not present in the documents, respond with: 'I'm not sure.'"
    )

    # Add HumanMessage to context
    messages.append(HumanMessage(content=combined_prompt))

    # Get response from model
    response = (model | parser).invoke(messages)

    # Add model response to context
    messages.append(AIMessage(content=response))

    # Print the response
    print("\n--- LLM Response ---")
    print(response)


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



#  AIzaSyBssOBIuYVVSGoHYjmpQB1nfd5P3rZFan8