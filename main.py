import os
import sys

from crewai import Agent, Task, Crew

from agents import llm

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


# ============================================================
# Configuration
# ============================================================

DOCUMENTS_FOLDER = "./documents"

VECTORSTORE_FOLDER = "./vectorstore"


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

        filename = file["filename"]

        file_path = file["file_path"]

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
# RAG Agent
# ============================================================

ragAgent = Agent(

    role="Document Question Answering Agent",

    goal="""
    Answer the user's question using the
    provided document information.
    """,

    backstory="""
    You are an expert document analyst.

    Answer only using the provided document context.

    Do not invent information.

    If the answer is not available in the context,
    clearly say that the information is not available
    in the documents.
    """,

    llm=llm

)


# ============================================================
# Show Thinking
# ============================================================

def show_thinking():

    print(
        "\nThinking...",
        end="",
        flush=True
    )


# ============================================================
# Remove Thinking
# ============================================================

def remove_thinking():

    sys.stdout.write(
        "\r" + (" " * 30) + "\r"
    )

    sys.stdout.flush()


# ============================================================
# Ask Question
# ============================================================

def ask_question(question):

    # --------------------------------------------
    # Show Thinking
    # --------------------------------------------

    show_thinking()


    # --------------------------------------------
    # FAISS files
    # --------------------------------------------

    vectorstore_index = os.path.join(

        VECTORSTORE_FOLDER,

        "index.faiss"

    )


    vectorstore_data = os.path.join(

        VECTORSTORE_FOLDER,

        "index.pkl"

    )


    # --------------------------------------------
    # Create FAISS on first question
    # --------------------------------------------

    if not (

        os.path.exists(
            vectorstore_index
        )

        and

        os.path.exists(
            vectorstore_data
        )

    ):

        document_items = read_all_documents()


        if not document_items:

            remove_thinking()

            return (
                "I couldn't read any documents. "
                "Please add PDF, CSV, Excel or DOCX "
                "files inside the documents folder."
            )


        create_vectorstore(
            document_items
        )


    # --------------------------------------------
    # Search FAISS
    # --------------------------------------------

    results = search_document(

        question,

        k=4

    )


    if not results:

        remove_thinking()

        return (
            "I couldn't find the answer "
            "in the documents."
        )


    # --------------------------------------------
    # Create Context
    # --------------------------------------------

    context = ""


    for document in results:

        context += (

            "\n\n"

            + document.page_content

        )


    # --------------------------------------------
    # Create Task
    # --------------------------------------------

    task = Task(

        description=f"""

        User Question:

        {question}


        Document Context:

        {context}


        Instructions:

        1. Answer the user's question.

        2. Use only the provided document context.

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

        A clear and accurate answer based only
        on the provided document context.

        """,

        agent=ragAgent

    )


    # --------------------------------------------
    # Create Crew
    # --------------------------------------------

    crew = Crew(

        agents=[
            ragAgent
        ],

        tasks=[
            task
        ]

    )


    # --------------------------------------------
    # Generate Answer
    # --------------------------------------------

    try:

        result = crew.kickoff()

        answer = result.raw


        # Remove Thinking
        remove_thinking()


        return answer


    except Exception:

        remove_thinking()


        return (
            "Sorry, I couldn't generate "
            "an answer right now."
        )


# ============================================================
# Application Header
# ============================================================

print(
    "\n======================================"
)

print(
    "       AI DOCUMENT SCANNER"
)

print(
    "======================================"
)

print(
    "Ask questions about your documents."
)

print(
    "Type 'exit' to close the application."
)

print(
    "======================================"
)


# ============================================================
# Main Loop
# ============================================================

while True:

    question = input(
        "\nYou: "
    ).strip()


    # --------------------------------------------
    # Exit
    # --------------------------------------------

    if question.lower() in [
        "exit",
        "quit"
    ]:

        print(
            "\nGoodbye! 👋"
        )

        break


    # --------------------------------------------
    # Empty question
    # --------------------------------------------

    if not question:

        continue


    # --------------------------------------------
    # Check Documents
    # --------------------------------------------

    files = get_documents()


    if not files:

        print(
            "\nAI: No documents are available. "
            "Please add PDF, CSV, Excel or DOCX "
            "files inside the documents folder."
        )

        continue


    # --------------------------------------------
    # Ask Question
    # --------------------------------------------

    answer = ask_question(
        question
    )


    # --------------------------------------------
    # Show Answer
    # --------------------------------------------

    print(
        f"AI: {answer}",
        flush=True
    )