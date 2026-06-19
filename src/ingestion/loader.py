"""
Extrai texto dos PDFs do corpus usando PyMuPDF (fitz).
Limpa artefatos comuns (headers, footers, números de página)
e preserva metadados de cada paper.
"""

import os
import re
import pymupdf as fitz  # PyMuPDF

def clean_text(text: str) -> str:
    """
    Remove artefatos comuns de PDFs acadêmicos:
    - Números de página isolados
    - Linhas com só espaços ou traços
    - Múltiplas quebras de linha seguidas
    - Hifenização no final de linha (re-une palavras)
    """
    # Re-une palavras hifenizadas no fim da linha (ex: "atten-\ntion" → "attention")
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Remove números de página isolados (linha só com número)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove linhas com só traços ou underscores
    text = re.sub(r"^\s*[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Colapsa múltiplas linhas em branco em uma só
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove espaços no início/fim
    text = text.strip()

    return text


def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extrai doc_id e título a partir do nome do arquivo.
    Exemplo: '01_attention_is_all_you_need.pdf'
             → {'doc_id': '01', 'title': 'attention is all you need'}
    """
    name = os.path.splitext(filename)[0]  # remove .pdf
    parts = name.split("_", 1)            # divide no primeiro underscore

    doc_id = parts[0] if len(parts) > 1 else "00"
    title = parts[1].replace("_", " ") if len(parts) > 1 else name

    return {"doc_id": doc_id, "title": title}

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrai todo o texto de um PDF página por página.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        text = page.get_text("text")  # extração simples em texto puro
        if text.strip():
            pages_text.append(text)

    doc.close()
    return "\n".join(pages_text)

def load_document(pdf_path: str) -> dict | None:
    """
    Carrega um único PDF e retorna um documento com texto e metadados.
    Retorna:
        {
            "doc_id": "01",
            "title": "attention is all you need",
            "filename": "01_attention_is_all_you_need.pdf",
            "filepath": "corpus/01_attention_is_all_you_need.pdf",
            "text": "...",
            "char_count": 12345,
            "page_count": 15
        }
    Retorna None se o texto não for extraível.
    """
    filename = os.path.basename(pdf_path)
    metadata = extract_metadata_from_filename(filename)

    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()

        raw_text = extract_text_from_pdf(pdf_path)
        clean = clean_text(raw_text)

        if len(clean.strip()) < 200:
            print(f"⚠️  [{metadata['doc_id']}] {filename} — texto insuficiente, pode precisar de OCR")
            return None

        return {
            "doc_id": metadata["doc_id"],
            "title": metadata["title"],
            "filename": filename,
            "filepath": pdf_path,
            "text": clean,
            "char_count": len(clean),
            "page_count": page_count,
        }

    except Exception as e:
        print(f"❌ [{metadata['doc_id']}] {filename} — erro: {e}")
        return None


def load_corpus(corpus_dir: str = "corpus") -> list[dict]:
    """
    Carrega todos os PDFs de uma pasta e retorna lista de documentos.
    Retorna lista de dicts, cada um com texto + metadados do paper.
    Papers com falha de extração são ignorados com aviso.
    """
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Pasta do corpus não encontrada: {corpus_dir}")

    pdf_files = sorted([
        f for f in os.listdir(corpus_dir)
        if f.endswith(".pdf")
    ])

    if not pdf_files:
        raise ValueError(f"Nenhum PDF encontrado em: {corpus_dir}")

    print(f"\n📂 Carregando corpus de '{corpus_dir}' ({len(pdf_files)} PDFs)...\n")

    documents = []
    for filename in pdf_files:
        path = os.path.join(corpus_dir, filename)
        print(f"📄 {filename}", end=" ... ", flush=True)
        doc = load_document(path)
        if doc:
            documents.append(doc)
            print(f"✅ {doc['char_count']:,} chars | {doc['page_count']} págs")

    print(f"\n✅ {len(documents)}/{len(pdf_files)} documentos carregados com sucesso.\n")
    return documents

# Teste rápido
if __name__ == "__main__":
    docs = load_corpus("corpus")
    print("Exemplo — primeiro documento:")
    print(f"  Título  : {docs[0]['title']}")
    print(f"  Chars   : {docs[0]['char_count']:,}")
    print(f"  Páginas : {docs[0]['page_count']}")
    print(f"  Trecho  : {docs[0]['text'][:300]}...")