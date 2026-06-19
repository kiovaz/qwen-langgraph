"""
Baixa papers do corpus direto do arXiv e valida que o texto é extraível (sem necessidade de OCR).
"""

import os
import time
import requests
import fitz  # PyMuPDF

CORPUS_DIR = "corpus"
SLEEP_BETWEEN_DOWNLOADS = 3

PAPERS = [
    # (doc_id, arxiv_id, filename_sem_extensao)
    ("01", "1706.03762", "01_attention_is_all_you_need"),
    ("02", "1810.04805", "02_bert"),
    ("03", "2001.08361", "03_scaling_laws_kaplan"),
    ("04", "2005.14165", "04_gpt3"),
    # ── Fine-tuning eficiente ─────────────────
    ("05", "2203.02155", "05_instructgpt_rlhf"),
    ("06", "2106.09685", "06_lora"),
    ("07", "2305.14314", "07_qlora"),
    ("08", "2307.09288", "08_llama2"),
    # ── Prompting & raciocínio ────────────────
    ("09", "2201.11903", "09_chain_of_thought"),
    ("21", "2402.07927", "21_prompt_engineering_survey"),
    # ── Agentes & ferramentas ─────────────────
    ("10", "2210.03629", "10_react"),
    ("11", "2302.04761", "11_toolformer"),
    ("12", "2305.16291", "12_voyager"),
    ("16", "2308.08155", "16_autogen"),
    # ── RAG ───────────────────────────────────
    ("13", "2005.11401", "13_rag_lewis"),
    ("14", "2312.10997", "14_rag_survey"),
    ("15", "2310.11511", "15_self_rag"),
    # ── Alinhamento ───────────────────────────
    ("17", "2212.08073", "17_constitutional_ai"),
    # ── Capacidades & arquiteturas modernas ───
    ("18", "2303.12528", "18_sparks_of_agi"),
    ("19", "2401.04088", "19_mixtral_moe"),
    # ── Avaliação & problemas ─────────────────
    ("20", "2309.01219", "20_hallucination_survey"),
    ("22", "2309.15217", "22_ragas"),
]

def is_text_extractable(pdf_path: str, min_chars: int = 200) -> tuple[bool, int]:
    """
    Abre o PDF e verifica se consegue extrair texto das primeiras páginas.
    Retorna (ok: bool, total_chars: int).
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        doc.close()
        total = len(text.strip())
        return total >= min_chars, total
    except Exception as e:
        print(f"⚠️ Erro ao abrir PDF: {e}")
        return False, 0


def download_paper(arxiv_id: str, output_path: str) -> bool:
    """
    Baixa um paper do arXiv pela URL direta do PDF.
    Retorna True se sucesso.
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; QwenLanggraphBot/1.0)"}

    try:
        response = requests.get(url, headers=headers, timeout=60, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except requests.exceptions.Timeout:
        print(f"❌ Timeout ao baixar {arxiv_id}")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP {e.response.status_code} ao baixar {arxiv_id}")
        return False
    except Exception as e:
        print(f"❌ Erro no download: {e}")
        return False

def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)

    total = len(PAPERS)
    ok_count = 0
    ocr_needed = []
    failed = []

    print(f"\n{'='*55}")
    print(f"Download do Corpus ({total} papers)")
    print(f"{'='*55}\n")

    for doc_id, arxiv_id, filename in PAPERS:
        output_path = os.path.join(CORPUS_DIR, f"{filename}.pdf")

        print(f"[{doc_id}/22] {filename}")
        print(f"arXiv: {arxiv_id}")

        # Pula se já existe
        if os.path.exists(output_path):
            print(f"⏭️  Já existe, pulando download.")
        else:
            print(f"⬇️  Baixando...", end=" ", flush=True)
            success = download_paper(arxiv_id, output_path)
            if not success:
                failed.append((doc_id, arxiv_id, filename))
                print()
                continue
            print("OK")
            time.sleep(SLEEP_BETWEEN_DOWNLOADS)

        # Valida extração de texto
        extractable, chars = is_text_extractable(output_path)
        if extractable:
            print(f"✅ Texto extraível ({chars:,} chars)")
            ok_count += 1
        else:
            print(f"⚠️  PRECISA DE OCR — texto insuficiente ({chars} chars)")
            ocr_needed.append(filename)

        print()

    print(f"{'='*55}")
    print(f"RESUMO")
    print(f"{'='*55}")
    print(f"✅ Prontos para ingestão : {ok_count}/{total}")

    if ocr_needed:
        print(f"\n⚠️  Precisam de OCR ({len(ocr_needed)}):")
        for name in ocr_needed:
            print(f"     - {name}.pdf")
        print(f"\n💡 Para aplicar OCR, use:")
        print(f"pip install ocrmypdf")
        print(f"ocrmypdf corpus/<arquivo>.pdf corpus/<arquivo>.pdf")

    if failed:
        print(f"\n❌ Falha no download ({len(failed)}):")
        for doc_id, arxiv_id, name in failed:
            print(f"     - [{doc_id}] {name} (arXiv: {arxiv_id})")
        print(f"\n💡 Tente baixar manualmente em:")
        for _, arxiv_id, _ in failed:
            print(f"https://arxiv.org/pdf/{arxiv_id}")

    if not ocr_needed and not failed:
        print(f"\n🎉 Corpus completo e pronto para ingestão!")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()