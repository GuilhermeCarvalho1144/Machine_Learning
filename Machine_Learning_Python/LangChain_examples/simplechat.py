from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import huggingface_pipeline
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DOC_PATH = "/mnt/1b0f3c80-0858-4ed5-a19d-c13144d4a615/Computarias/Machine_Learning/Machine_Learning_Python/Movie_IMDB_dataset/test.txt"

loader = TextLoader(DOC_PATH)
document = loader.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
docs = text_spliter.split_documents(document)

embed = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(docs, embed)

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    top_k=50,
)
llm = huggingface_pipeline.HuggingFacePipeline(pipeline=generator)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or your chosen chain type
    retriever=vectorstore.as_retriever(),
)


def main():
    print("SIMPLE CHATBOT")
    print("type 'exit' to quit")
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        ans = qa_chain.invoke(query)
        print("\n Answer: ", ans)


if __name__ == "__main__":
    main()
