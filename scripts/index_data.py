"""
Script to index data (PDF/TXT/MD) into a FAISS vector store for PrepAI.
This version avoids concurrency by manually iterating over files to fix segmentation fault issues.
"""

import os
import argparse
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from vector_store.data_retriever import DataRetriever
from config.settings import settings
import glob


def main(data_dir: str) -> None:
    """
    Load PDF, TXT, and MD files from `data_dir` and index them into FAISS.
    A single-threaded approach to mitigate segmentation faults.

    Args:
        data_dir: Path to the directory containing PDF, text, or markdown files.
    """
    print(settings.VECTOR_STORE_PATH)
    retriever = DataRetriever()
    # -------------------------------
    # 3) Load MD files (single-threaded)
    # -------------------------------
    md_docs = []
    md_files = glob.glob(os.path.join(data_dir, "*.md"))
    print(md_files)
    for md_file in md_files:
        loader = TextLoader(md_file)
        try:
            md_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {md_file}: {e}")

    # Combine all documents into a single list
    all_docs = md_docs
    print(len(all_docs))

    # Extract just the text content
    documents_text = [
        doc.page_content.strip()
        for doc in all_docs
        if doc.page_content and doc.page_content.strip()
    ]

    print(len(documents_text))

    try:
        if documents_text:
            retriever.add_documents(documents_text)
            print(
                f"Successfully indexed {len(documents_text)} documents into FAISS at "
                f"{settings.VECTOR_STORE_PATH}"
            )
        else:
            print("No documents found to index.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index data (PDF/TXT/MD) into a FAISS vector store for PrepAI."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=settings.DATA_PATH,
        help="Directory containing PDFs, text, or markdown files."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    main(args.data_dir)
