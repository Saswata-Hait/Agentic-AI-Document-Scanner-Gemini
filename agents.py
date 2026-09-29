from crewai import Agent, LLM

from tools import (
    readPDF,
    readCSV,
    readExcel,
    readWord
)

from dotenv import load_dotenv

import os


# ============================================================
# Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# Connect to Gemini
# ============================================================

llm = LLM(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)


# ============================================================
# PDF Reader Agent
# ============================================================

pdfReaderAgent = Agent(

    role="PDF Document Reader",

    goal="""
    Read PDF documents and accurately extract
    useful information from them.
    """,

    backstory="""
    You are an expert PDF document reader.

    You carefully read PDF documents and extract
    useful information from every page.

    You provide accurate document content for
    the RAG system.
    """,

    tools=[
        readPDF
    ],

    llm=llm

)


# ============================================================
# CSV Reader Agent
# ============================================================

csvReaderAgent = Agent(

    role="CSV Document Reader",

    goal="""
    Read CSV files and accurately extract
    structured tabular information.
    """,

    backstory="""
    You are an expert CSV data reader.

    You understand rows, columns and tabular
    information stored inside CSV files.

    You provide the CSV content accurately
    for the RAG system.
    """,

    tools=[
        readCSV
    ],

    llm=llm

)


# ============================================================
# Excel Reader Agent
# ============================================================

excelReaderAgent = Agent(

    role="Excel Document Reader",

    goal="""
    Read Excel files and extract information
    from all worksheets.
    """,

    backstory="""
    You are an expert Excel document reader.

    You understand worksheets, rows, columns,
    tables and structured spreadsheet information.

    You carefully process every worksheet in
    an Excel file.
    """,

    tools=[
        readExcel
    ],

    llm=llm

)


# ============================================================
# Word Reader Agent
# ============================================================

wordReaderAgent = Agent(

    role="Word Document Reader",

    goal="""
    Read DOCX documents and accurately extract
    their useful information.
    """,

    backstory="""
    You are an expert Microsoft Word document reader.

    You extract information from paragraphs
    and tables inside DOCX documents accurately.

    You provide the extracted content for
    the RAG system.
    """,

    tools=[
        readWord
    ],

    llm=llm

)