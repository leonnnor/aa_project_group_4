# DO NOT CHANGE THE NAME OF THIS FILE
import os
from pathlib import Path


# THE FOLLOWING TWO LINES SHOULD NOT BE CHANGED.
# All generated files by your agent should be stored in a folder given by this
# STORAGE_DIR environmental parameter.
# Your agent should use the session ID given by the environment parameter
# SESSION_ID to select conversation memory.
STORAGE_DIR = Path(os.environ.get("AGENT_STORAGE_DIR", "./agent_data")).resolve()
SESSION_ID = os.environ.get("AGENT_SESSION_ID", "default")


def answer(prompt: str) -> str:
    """Return the final answer to ONE user message, as a nonempty string.

    Implement your final agent features here. Repeated calls in the same process
    should be considered to belong to the same conversation. Save durable memory
    before returning and restore it on the next launch with the same STORAGE_DIR
    and SESSION_ID.

    Do not call input(), return a generator, or require the caller to supply
    history. Raise an exception on failure instead of returning an error as an
    apparently successful answer. Printing diagnostic logs is allowed.
    """

    # YOUR IMPLEMENTATION STARTS HERE.

    return "MOCK ANSWER: replace this response with your agent's answer."


def reset() -> None:
    """Clear all the conversation history memory; return None.

    A call to this function should only delete the conversation history.
    The document index (RAG) should be maintained.
    """

    # YOUR IMPLEMENTATION STARTS HERE.

    return None


def ingest(folder: str) -> int:
    """Replace the shared knowledge base using documents in this folder.

    `folder` is an absolute path to a read-only source folder. Required baseline:
    recursively load UTF-8 .txt files. Additional formats are optional. Do not
    modify the originals. Finish loading, splitting, embedding, indexing, and
    saving before returning the number of successfully indexed SOURCE FILES
    (not chunks). The saved index must survive process restart. Repeating this
    call replaces the old collection without duplicating documents and must not
    delete conversation memory. Raise an exception on failure. An empty folder
    returns 0 and replaces the index with an empty collection.

    File discovery and text loading are provided below. Complete the processing
    and persistence section before this function can report successful ingestion.
    """
    source_folder = Path(folder).expanduser().resolve(strict=True)
    if not source_folder.is_dir():
        raise NotADirectoryError(f"Expected a document folder: {source_folder}")

    # Discover files recursively in a predictable order. Accept .txt and .TXT;
    # ignore other formats. Paths with spaces and non-ASCII names work normally.
    paths = sorted(
        path for path in source_folder.rglob("*")
        if path.is_file() and path.suffix.lower() == ".txt"
    )

    documents = []
    for path in paths:
        # utf-8-sig also accepts ordinary UTF-8 and removes an optional BOM.
        # Let read/decoding errors propagate: silently skipping a file would
        # make the instructor's knowledge base incomplete.
        text = path.read_text(encoding="utf-8-sig")
        documents.append({
            "text": text,
            # Preserve a portable source label for citations and chunk metadata.
            # Relative paths distinguish equally named files in subfolders.
            "source": path.relative_to(source_folder).as_posix(),
        })

    # YOUR IMPLEMENTATION STARTS HERE.

    # `documents` now contains one dictionary per source file, for example:
    # {"text": "I started a new project called ...", "source": "projects/new_project.txt"}
    #
    # 1. Prepare a replacement document index beneath STORAGE_DIR.
    # 2. For each document in `documents`, process document["text"], split it
    #    into chunks, generate embeddings, and store the chunks/embeddings.
    #    Include document["source"] into each chunk's metadata for citations.
    # 3. Persist the finished index so another process can reopen it. Replace
    #    the previous collection, including when `documents` is empty, without
    #    erasing conversation memory. Do not append duplicate collections.
    # 4. Only after successful indexing, replace the exception below with:
    #    return len(documents)
    
    raise NotImplementedError("Add document processing, chunking, embeddings, and index storage.")
