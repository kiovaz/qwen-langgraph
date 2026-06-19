"""
Divide os documentos extraídos pelo loader.py em chunks menores
para indexação no ChromaDB.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 4500
CHUNK_OVERLAP = 400

def chunk_documents(documents: list[dict]) -> list[dict]:

    """
    Recebe lista de documentos do loader e retorna lista de chunks.

    Cada chunk tem:
        {
            "chunk_id":   "01_0003",        # doc_id + número do chunk
            "doc_id":     "01",             # id do paper de origem
            "title":      "attention is all you need",
            "filename":   "01_attention_is_all_you_need.pdf",
            "text":       "...",            # trecho do texto
            "chunk_index": 3,              # posição dentro do paper
            "total_chunks": 42,            # total de chunks desse paper
        }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in documents:
        raw_chunks = splitter.split_text(doc["text"])
        total = len(raw_chunks)

        for i, chunk_text in enumerate(raw_chunks):
            chunk = {
                "chunk_id":    f"{doc['doc_id']}_{i:04d}",
                "doc_id":      doc["doc_id"],
                "title":       doc["title"],
                "filename":    doc["filename"],
                "text":        chunk_text,
                "chunk_index": i,
                "total_chunks": total,
            }
            all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.ingestion.loader import load_corpus

    docs = load_corpus("corpus")
    chunks = chunk_documents(docs)

    print(f"📄 Documentos carregados : {len(docs)}")
    print(f"🔪 Total de chunks       : {len(chunks)}")
    print(f"📊 Média por documento   : {len(chunks) / len(docs):.1f} chunks\n")

    print("Exemplo — primeiro chunk:")
    c = chunks[0]
    print(f"  chunk_id     : {c['chunk_id']}")
    print(f"  doc_id       : {c['doc_id']}")
    print(f"  title        : {c['title']}")
    print(f"  chunk_index  : {c['chunk_index']} / {c['total_chunks']}")
    print(f"  chars        : {len(c['text'])}")
    print(f"  trecho       : {c['text'][:200]}...")