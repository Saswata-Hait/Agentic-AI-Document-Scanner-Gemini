import os

import streamlit as st

from tools import (
    readPDF,
    readCSV,
    readExcel,
    readWord
)

from rag import (
    create_vectorstore,
    search_document
)

from agents import llm

from crewai import Agent, Task, Crew


# ============================================================
# Configuration
# ============================================================

DOCUMENTS_FOLDER = "./documents"

VECTORSTORE_FOLDER = "./vectorstore"


# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(

    page_title="AI Document Scanner",

    page_icon="",

    layout="centered"

)


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# Create Documents Folder
# ============================================================

if not os.path.exists(
    DOCUMENTS_FOLDER
):

    os.makedirs(
        DOCUMENTS_FOLDER
    )


# ============================================================
# Get Documents
# ============================================================

def get_documents():

    documents = []

    for filename in os.listdir(
        DOCUMENTS_FOLDER
    ):

        file_path = os.path.join(

            DOCUMENTS_FOLDER,

            filename

        )


        if not os.path.isfile(
            file_path
        ):

            continue


        extension = (

            filename
            .lower()
            .split(".")[-1]

        )


        if extension in [

            "pdf",
            "csv",
            "xlsx",
            "xls",
            "docx"

        ]:

            documents.append({

                "filename": filename,

                "file_path": file_path,

                "extension": extension

            })


    return documents


# ============================================================
# Read All Documents
# ============================================================

def read_all_documents():

    document_items = []

    files = get_documents()


    for file in files:

        file_path = file["file_path"]

        filename = file["filename"]

        extension = file["extension"]


        try:

            if extension == "pdf":

                text = readPDF.run(
                    file_path
                )


            elif extension == "csv":

                text = readCSV.run(
                    file_path
                )


            elif extension in [
                "xlsx",
                "xls"
            ]:

                text = readExcel.run(
                    file_path
                )


            elif extension == "docx":

                text = readWord.run(
                    file_path
                )


            else:

                continue


            if (

                text

                and

                not text.startswith("Error")

            ):

                document_items.append({

                    "text": text,

                    "source": filename,

                    "file_type": extension

                })


        except Exception:

            continue


    return document_items


# ============================================================
# Header
# ============================================================

st.title(
    "AI Document Scanner"
)

st.caption(
    "Ask anything about your documents."
)


# ============================================================
# Display Previous Messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# Chat Input
# ============================================================

question = st.chat_input(
    "Ask something..."
)


# ============================================================
# User Sends Question
# ============================================================

if question:

    # --------------------------------------------
    # Save User Message
    # --------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # --------------------------------------------
    # Display User Message
    # --------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------
    # Check Documents
    # --------------------------------------------

    files = get_documents()


    if not files:

        answer = (

            "No documents are available. "

            "Please add PDF, CSV, Excel or DOCX "

            "files to the documents folder."

        )


        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )


        st.session_state.messages.append({

            "role": "assistant",

            "content": answer

        })


        st.stop()


    # ========================================================
    # Assistant Response
    # ========================================================

    with st.chat_message(
        "assistant"
    ):


        # --------------------------------------------
        # Thinking Loader
        # --------------------------------------------

        with st.spinner(
            "Thinking..."
        ):


            # ========================================
            # Check FAISS
            # ========================================

            vectorstore_index = os.path.join(

                VECTORSTORE_FOLDER,

                "index.faiss"

            )


            vectorstore_data = os.path.join(

                VECTORSTORE_FOLDER,

                "index.pkl"

            )


            # ========================================
            # Create FAISS if not available
            # ========================================

            if not (

                os.path.exists(
                    vectorstore_index
                )

                and

                os.path.exists(
                    vectorstore_data
                )

            ):

                document_items = (
                    read_all_documents()
                )


                if not document_items:

                    answer = (
                        "I couldn't read the documents."
                    )


                    st.markdown(
                        answer
                    )


                    st.session_state.messages.append({

                        "role": "assistant",

                        "content": answer

                    })


                    st.stop()


                create_vectorstore(
                    document_items
                )


            # ========================================
            # Search FAISS
            # ========================================

            results = search_document(

                question,

                k=4

            )


            # ========================================
            # No Results
            # ========================================

            if not results:

                answer = (

                    "I couldn't find the answer "

                    "in the documents."

                )


            else:

                # ------------------------------------
                # Create Context
                # ------------------------------------

                context = ""


                for document in results:

                    context += (

                        "\n\n"

                        + document.page_content

                    )


                # ------------------------------------
                # RAG Agent
                # ------------------------------------

                ragAgent = Agent(

                    role=(
                        "Document Question "
                        "Answering Agent"
                    ),

                    goal="""
                    Answer the user's question using
                    the provided document information.
                    """,

                    backstory="""
                    You are an expert document analyst.

                    Answer only using the provided
                    document context.

                    Do not invent information.

                    If the answer is not available
                    in the context, clearly say that
                    the information is not available
                    in the documents.
                    """,

                    llm=llm

                )


                # ------------------------------------
                # Task
                # ------------------------------------

                task = Task(

                    description=f"""

                    User Question:

                    {question}


                    Document Context:

                    {context}


                    Instructions:

                    1. Answer the user's question.

                    2. Use only the provided
                       document context.

                    3. Do not invent information.

                    4. Give a clear and concise answer.

                    5. Do not mention filenames,
                       sources or document metadata.

                    6. If the answer is not available,
                       say:

                    "The information is not available
                    in the documents."

                    """,

                    expected_output="""

                    A clear and accurate answer based
                    only on the provided document context.

                    """,

                    agent=ragAgent

                )


                # ------------------------------------
                # Crew
                # ------------------------------------

                crew = Crew(

                    agents=[
                        ragAgent
                    ],

                    tasks=[
                        task
                    ]

                )


                # ------------------------------------
                # Generate Answer
                # ------------------------------------

                try:

                    result = crew.kickoff()

                    answer = result.raw


                except Exception:

                    answer = (

                        "Sorry, I couldn't generate "

                        "an answer right now."

                    )


            # ========================================
            # Display Answer
            # ========================================

            st.markdown(
                answer
            )


    # ========================================================
    # Save Assistant Message
    # ========================================================

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })