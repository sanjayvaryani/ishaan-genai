#importing all the necessary libraries
import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
#function to read the documents
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
#mention the directory where the documents are kept
doc=read_doc('doc/')
#len(doc)
#function to make chunks of the documents uploaded
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs
#storing the chunks in an object documents
documents=chunk_data(docs=doc)
#len(documents)

## Embedding Technique Of OPENAI
embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

from langchain_pinecone import PineconeVectorStore

#uploading the embeddings into Pinecone vectordatabase
#IMP change the index name
#CREATE an index using pinecone console and mention that name in 'index_name' 
vectorstore_from_docs = PineconeVectorStore.from_documents(
    documents,
    index_name='ishaan-genai',
    embedding=embeddings
)


#following code is for q&a without making a UI. Uncomment and run if you want to check else ignore and go to the file app.py

# ## Cosine Similarity Retreive Results from VectorDB
# def retrieve_query(query,k=2):
#     matching_results=vectorstore_from_docs.similarity_search(query,k=k)
#     return matching_results

# from langchain.chains.question_answering import load_qa_chain
# from langchain import OpenAI

# llm=OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.5)
# chain=load_qa_chain(llm,chain_type="stuff")

# ## Search answers from VectorDB
# def retrieve_answers(query):
#     doc_search=retrieve_query(query)
#     print(doc_search)
#     response=chain.run(input_documents=doc_search,question=query)
#     return response
