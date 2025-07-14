#NLP Pipeline for regulatory documents summarization 
import os 
import json 
import re
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
#using 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from transformers import pipeline
from langchain_groq.chat_models import ChatGroq
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("API_KEY")

#load the data from the data folder and split the pdf into chunks
def load_and_split(filepath):
    all_docs = []
    for file in os.listdir(filepath):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(filepath, file))
            docs = loader.load()
            all_docs.extend(docs)

    if not all_docs:
        print("No readable PDFs found!")
        return []

    print(f"Loaded {len(all_docs)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

# summarize the text using a summarization model and ChatGroq
def summarize_documents(split_docs, temperature=0.2):
    llm = ChatGroq(model="llama-3.3-70b-versatile",              
    temperature = temperature)

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    summary = chain.run(split_docs)
    return summary


#extract obligations from the summary using a regex pattern
def extract_obligations(summary_text):
    obligation_sentences = []
    sentences = re.split(r'(?<=[.?!])\s+', summary_text)
    for sentence in sentences:
        if re.search(r'\b(must|shall|required|ensure|comply|prohibited)\b', sentence, re.IGNORECASE):
            obligation_sentences.append(sentence.strip())
    return obligation_sentences

#format the obligations into a structured JSON format
def format_obligations(obligations, source="NLP Summarizer"):
    formatted_obligations = []
    for obligation in obligations:
        formatted_obligations.append({"obligation": obligation, 
                                      "source": source, 
                                      "type": "obligation"})
    return json.dumps(formatted_obligations, indent=4)

#full pipeline to run the summarization and obligation extraction
def regula_ai_nlp_pipeline(filepath, source_name="NLP Summarizer"):
    split_docs = load_and_split(filepath)
    summary = summarize_documents(split_docs)
    print("\nRaw Summary Output:\n", summary)  # Add this
    obligations = extract_obligations(summary)
    print("\nExtracted Obligation Sentences:\n", obligations)  # Add this
    formatted_obligations = format_obligations(obligations, source = source_name)
    return formatted_obligations


#testing the pipeline with the data folder
if __name__ == "__main__":
    data_folder = r"C:\Projects_ML\Regula-AI\NLP Summarizer\Data"
    source_name = "NLP Summarizer"
    obligations_json = regula_ai_nlp_pipeline(data_folder, source_name)
    print(obligations_json)

#example to run the script to summarize a PDF and extract obligations
# python mvp_nlp_summarizer.py
