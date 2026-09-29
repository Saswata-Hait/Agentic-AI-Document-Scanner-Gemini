import warnings

warnings.filterwarnings(
    "ignore",
    message=".*HuggingFaceEmbeddings.*deprecated.*"
)
import os
import shutil

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


VECTORSTORE_PATH = "./vectorstore"


# ============================================================
# Hugging Face Embeddings
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# Create Documents
# ============================================================

def create_documents(document_items):

    documents = []

    for item in document_items:

        document = Document(
            page_content=item["text"],
            metadata={
                "source": item["source"],
                "file_type": item["file_type"]
            }
        )

        documents.append(document)

    return documents


# ============================================================
# Split Documents
# ============================================================

def split_documents(document_items):

    documents = create_documents(
        document_items
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(
        documents
    )


# ============================================================
# Create FAISS Vectorstore
# ============================================================

def create_vectorstore(document_items):

    chunks = split_documents(
        document_items
    )

    if not chunks:
        return 0

    if os.path.exists(VECTORSTORE_PATH):

        shutil.rmtree(
            VECTORSTORE_PATH
        )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        VECTORSTORE_PATH
    )

    return len(chunks)


# ============================================================
# Load FAISS Vectorstore
# ============================================================

def load_vectorstore():

    index_file = os.path.join(
        VECTORSTORE_PATH,
        "index.faiss"
    )

    data_file = os.path.join(
        VECTORSTORE_PATH,
        "index.pkl"
    )

    if not (
        os.path.exists(index_file)
        and
        os.path.exists(data_file)
    ):
        return None

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# ============================================================
# Search Documents
# ============================================================

def search_document(
    question,
    k=4
):

    vectorstore = load_vectorstore()

    if vectorstore is None:
        return []

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results