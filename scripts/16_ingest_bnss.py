"""
16_ingest_bnss.py
=================
Ingests BNSS 2023 (Bharatiya Nagarik Suraksha Sanhita) into
the existing statute FAISS index alongside IPC and BNS.

The BNSS 2023 has 531 sections covering criminal procedure
(replacing CrPC 1973).

Usage:
    python scripts/16_ingest_bnss.py --hardcoded --index_dir indexes/
    python scripts/16_ingest_bnss.py --pdf data/bnss_2023.pdf --index_dir indexes/
    python scripts/16_ingest_bnss.py --text data/bnss_2023.txt --index_dir indexes/
"""

import json, re, time, argparse, sys
import numpy as np
import faiss, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))


BNSS_SECTION_PATTERN = re.compile(
    r'(?:^|\n)\s*(\d+)\s*\.\s*([^\n]{5,120})\n(.*?)(?=\n\s*\d+\s*\.|$)',
    re.DOTALL
)

# Key BNSS sections with their CrPC equivalents for cross-referencing
BNSS_TO_CRPC_MAP = {
    "480": "437",   # Bail in bailable offences
    "481": "437A",  # Before sureties
    "482": "438",   # Anticipatory bail — KEY SECTION
    "483": "439",   # Special powers of High Court re bail
    "484": "439A",
    "173": "154",   # FIR
    "176": "157",   # Report to Magistrate
    "187": "167",   # Remand
    "35" : "41",    # Arrest without warrant
    "38" : "46",    # How arrest is made
    "43" : "50",    # Person arrested to be informed of grounds
    "47" : "55",    # Health of arrested person
    "193": "173",   # Police report / chargesheet
    "230": "207",   # Supply of documents to accused
    "251": "228",   # Framing of charge
    "358": "304",   # Legal aid to accused
    "528": "482",   # High Court inherent powers
}


def parse_bnss_from_text(text: str) -> list[dict]:
    """Parse BNSS sections from plain text."""
    sections = []
    # Split on section number patterns
    parts = re.split(r'\n(?=\d{1,3}\.\s+[A-Z])', text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        m = re.match(r'^(\d{1,3})\.\s+(.+?)[\.\u2014\-]\s*(.*)', part, re.DOTALL)
        if not m:
            # Try simpler pattern
            m2 = re.match(r'^(\d{1,3})\.\s+(.+)', part, re.DOTALL)
            if m2:
                sec_num   = m2.group(1).strip()
                remainder = m2.group(2).strip()
                # First line = title, rest = body
                lines      = remainder.split('\n', 1)
                sec_title  = lines[0].strip()[:120]
                sec_text   = lines[1].strip() if len(lines) > 1 else sec_title
            else:
                continue
        else:
            sec_num   = m.group(1).strip()
            sec_title = m.group(2).strip()[:120]
            sec_text  = m.group(3).strip()

        if len(sec_text) < 20:
            sec_text = part  # fallback to full text

        crpc_equiv = BNSS_TO_CRPC_MAP.get(sec_num, "")

        sections.append({
            "source"       : "BNSS 2023",
            "section_num"  : sec_num,
            "section_title": sec_title,
            "text"         : f"BNSS 2023 Section {sec_num} — {sec_title}\n{sec_text}",
            "act"          : "Bharatiya Nagarik Suraksha Sanhita 2023",
            "crpc_equivalent": crpc_equiv,
            "chunk_type"   : "statute",
        })

    print(f"Parsed {len(sections)} BNSS sections")
    return sections


def parse_bnss_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF then parse."""
    try:
        import PyPDF2
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return parse_bnss_from_text(text)
    except ImportError:
        print("PyPDF2 not installed. Run: pip install PyPDF2")
        raise


def embed_and_append_to_index(
    sections   : list[dict],
    index_dir  : Path,
    model_name : str = "intfloat/e5-large-v2",
    batch_size : int = 32,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
):
    """Embed new sections and append them to the existing statute FAISS index."""

    config = json.loads((index_dir / "index_config.json").read_text())
    model_name     = config.get("model_name",   model_name)
    query_prefix   = config.get("query_prefix", query_prefix)
    normalize      = config.get("normalize",    True)

    # Check VRAM before loading
    device = "cpu"
    if torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        if free_vram >= 1.8:
            device = "cuda"
        else:
            print(f"  Only {free_vram:.1f}GB VRAM free — using CPU.")

    print(f"Loading {model_name} on {device}...")
    try:
        model = SentenceTransformer(
            model_name, device=device,
            model_kwargs={"use_safetensors": False}
        )
    except Exception:
        print("  Fallback: loading without use_safetensors=False...")
        model = SentenceTransformer(model_name, device="cpu")

    # Load existing statute index
    statute_index_path = index_dir / "faiss_statutes.index"
    statute_meta_path  = index_dir / "faiss_statutes_metadata.jsonl"

    existing_index = faiss.read_index(str(statute_index_path))
    print(f"Existing statute index: {existing_index.ntotal} vectors")

    # Check for duplicates — don't add sections already indexed
    existing_meta = []
    with open(statute_meta_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                existing_meta.append(json.loads(line))

    existing_keys = {
        (m.get("source",""), m.get("section_num",""))
        for m in existing_meta
    }

    new_sections = [
        s for s in sections
        if (s["source"], s["section_num"]) not in existing_keys
    ]
    print(f"New sections to add: {len(new_sections)} "
          f"(skipping {len(sections)-len(new_sections)} duplicates)")

    if not new_sections:
        print("Nothing to add — all sections already indexed.")
        return

    # Embed in batches
    texts = [
        f"{passage_prefix}{s['text']}" if passage_prefix else s["text"]
        for s in new_sections
    ]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/"
              f"{(len(texts)-1)//batch_size + 1} "
              f"({len(batch)} sections)...")
        with torch.no_grad():
            embs = model.encode(
                batch,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
        all_embeddings.append(embs)

    new_embeddings = np.vstack(all_embeddings)

    # Append to FAISS index
    existing_index.add(new_embeddings)
    faiss.write_index(existing_index, str(statute_index_path))
    print(f"Statute index updated: {existing_index.ntotal} total vectors")

    # Append metadata
    with open(statute_meta_path, "a", encoding="utf-8") as f:
        for section in new_sections:
            f.write(json.dumps(section, ensure_ascii=False) + "\n")

    print(f"Metadata updated: added {len(new_sections)} entries")
    print(f"Done. BNSS 2023 is now in the index.")


def create_bnss_from_known_sections() -> list[dict]:
    """
    Fallback: create index entries for the most critical BNSS sections
    from hardcoded text if no PDF is available.
    These are the sections most commonly queried in Indian criminal law.
    """
    CRITICAL_BNSS = [
        {
            "section_num"  : "482",
            "section_title": "Direction for grant of bail to person apprehending arrest",
            "text"         : (
                "BNSS 2023 Section 482 — Direction for grant of bail to person "
                "apprehending arrest (Anticipatory Bail)\n"
                "When any person has reason to believe that he may be arrested on "
                "an accusation of having committed a non-bailable offence, he may "
                "apply to the High Court or the Court of Session for a direction "
                "under this section; and that Court may, if it thinks fit, direct "
                "that in the event of such arrest, he shall be released on bail. "
                "The Court may impose conditions including: (i) making oneself "
                "available for interrogation, (ii) not making inducement or threat "
                "to any person acquainted with the facts, (iii) not leaving India "
                "without permission. "
                "This is the BNSS equivalent of Section 438 CrPC 1973."
            ),
            "crpc_equivalent": "438",
        },
        {
            "section_num"  : "480",
            "section_title": "Bail in bailable offences",
            "text"         : (
                "BNSS 2023 Section 480 — Bail in bailable offences\n"
                "When any person other than a person accused of a non-bailable "
                "offence is arrested or detained without warrant by an officer "
                "in charge of a police station, or appears or is brought before "
                "a Court, and is prepared at any time while in the custody of "
                "such officer or at any stage of the proceeding before such Court "
                "to give bail, such person shall be released on bail. "
                "A person accused of a bailable offence may be released on bail "
                "at any stage. "
                "This is the BNSS equivalent of Section 437 CrPC 1973."
            ),
            "crpc_equivalent": "437",
        },
        {
            "section_num"  : "481",
            "section_title": "Bail in non-bailable offences",
            "text"         : (
                "BNSS 2023 Section 481 — Bail in non-bailable offences\n"
                "When any person accused of, or suspected of, the commission of "
                "any non-bailable offence is arrested or detained without warrant "
                "by an officer in charge of a police station or appears or is "
                "brought before a Court other than the High Court or Court of "
                "Session, he may be released on bail. However, such person shall "
                "not be so released if there appear reasonable grounds for "
                "believing that he has been guilty of an offence punishable with "
                "death or imprisonment for life. "
                "The Court shall consider the nature and gravity of the accusation, "
                "antecedents of the applicant, and possibility of his fleeing "
                "from justice. "
                "This is the BNSS equivalent of Section 437 CrPC 1973 (non-bailable part)."
            ),
            "crpc_equivalent": "437",
        },
        {
            "section_num"  : "483",
            "section_title": "Special powers of High Court or Court of Session regarding bail",
            "text"         : (
                "BNSS 2023 Section 483 — Special powers of High Court or Court "
                "of Session regarding bail\n"
                "A High Court or Court of Session may direct that any person "
                "accused of an offence and in custody be released on bail, and "
                "if the offence is of the nature specified in sub-section (3) "
                "of section 480, may impose any condition which it considers "
                "necessary for the purposes mentioned in that sub-section. "
                "The High Court or Court of Session may also direct that bail "
                "granted under section 480 be cancelled if the accused misuses "
                "the liberty. "
                "This is the BNSS equivalent of Section 439 CrPC 1973."
            ),
            "crpc_equivalent": "439",
        },
        {
            "section_num"  : "173",
            "section_title": "Information in cognizable cases — FIR",
            "text"         : (
                "BNSS 2023 Section 173 — Information in cognizable cases (First Information Report)\n"
                "Every information relating to the commission of a cognizable "
                "offence, if given orally to an officer in charge of a police "
                "station, shall be reduced to writing by him or under his "
                "direction, and be read over to the informant; and every such "
                "information, whether given in writing or reduced to writing as "
                "aforesaid, shall be signed by the person giving it. "
                "A copy of the information as recorded shall be given forthwith, "
                "free of cost, to the informant. "
                "The information may also be given by electronic communication. "
                "This is the BNSS equivalent of Section 154 CrPC 1973 (FIR)."
            ),
            "crpc_equivalent": "154",
        },
        {
            "section_num"  : "35",
            "section_title": "Arrest of persons — without warrant",
            "text"         : (
                "BNSS 2023 Section 35 — When police may arrest without warrant\n"
                "Any police officer may without an order from a Magistrate and "
                "without a warrant, arrest any person who has been concerned in "
                "any cognizable offence, or against whom a reasonable complaint "
                "has been made, or credible information has been received, or "
                "a reasonable suspicion exists of his having been so concerned. "
                "For offences punishable with imprisonment up to 3 years, police "
                "shall not arrest if the person has a fixed place of abode and "
                "is not likely to commit further offence. "
                "This is the BNSS equivalent of Section 41 CrPC 1973."
            ),
            "crpc_equivalent": "41",
        },
        {
            "section_num"  : "187",
            "section_title": "Remand — procedure when investigation cannot be completed in 24 hours",
            "text"         : (
                "BNSS 2023 Section 187 — Procedure when investigation cannot "
                "be completed in twenty-four hours\n"
                "Whenever any person is arrested and detained in custody, and "
                "it appears that the investigation cannot be completed within "
                "the period of twenty-four hours fixed by Section 58, and there "
                "are grounds for believing that the accusation or information is "
                "well-founded, the officer in charge of the police station shall "
                "transmit to the nearest Judicial Magistrate a copy of the "
                "entries in the diary and shall at the same time forward the "
                "accused to such Magistrate. "
                "The Magistrate may authorise detention in police custody for "
                "up to fifteen days, and judicial custody up to sixty or ninety "
                "days depending on severity. "
                "This is the BNSS equivalent of Section 167 CrPC 1973 (remand)."
            ),
            "crpc_equivalent": "167",
        },
        {
            "section_num"  : "528",
            "section_title": "High Court inherent powers",
            "text"         : (
                "BNSS 2023 Section 528 — Saving of inherent powers of High Court\n"
                "Nothing in this Sanhita shall be deemed to limit or affect the "
                "inherent powers of the High Court to make such orders as may be "
                "necessary to give effect to any order under this Sanhita, or to "
                "prevent abuse of the process of any Court or otherwise to secure "
                "the ends of justice. "
                "This is the BNSS equivalent of Section 482 CrPC 1973."
            ),
            "crpc_equivalent": "482",
        },
        {
            "section_num"  : "43",
            "section_title": "Person arrested to be informed of grounds of arrest",
            "text"         : (
                "BNSS 2023 Section 43 — Person arrested to be informed of grounds of arrest "
                "and of right to bail\n"
                "Every police officer or other person arresting any person without "
                "warrant shall forthwith communicate to him full particulars of "
                "the offence for which he is arrested or other grounds for such "
                "arrest. Where a police officer arrests without warrant any person "
                "other than a person accused of a non-bailable offence, he shall "
                "inform the person arrested that he is entitled to be released on bail. "
                "This is the BNSS equivalent of Section 50 CrPC 1973."
            ),
            "crpc_equivalent": "50",
        },
        {
            "section_num"  : "193",
            "section_title": "Report of police officer on completion of investigation — Chargesheet",
            "text"         : (
                "BNSS 2023 Section 193 — Report of police officer on completion "
                "of investigation (Chargesheet / Final Report)\n"
                "Every investigation shall be completed without unnecessary delay "
                "and the report shall be forwarded to the Magistrate. If upon "
                "investigation it appears that there is sufficient evidence or "
                "reasonable ground, the officer in charge of the police station "
                "shall forward to a Magistrate a report in the form prescribed "
                "by the State Government, stating the names of the parties, the "
                "nature of the information, and the names of persons who appear "
                "to be acquainted with the circumstances of the case. "
                "This is the BNSS equivalent of Section 173 CrPC 1973 (chargesheet)."
            ),
            "crpc_equivalent": "173",
        },
        {
            "section_num"  : "358",
            "section_title": "Legal aid to accused at State expense",
            "text"         : (
                "BNSS 2023 Section 358 — Legal aid to accused at State expense\n"
                "Where, in a trial before the Court of Session, the accused is not "
                "represented by a pleader, and where it appears to the Court that "
                "the accused has not sufficient means to engage a pleader, the "
                "Court shall assign a pleader for his defence at the expense of "
                "the State. "
                "This is the BNSS equivalent of Section 304 CrPC 1973."
            ),
            "crpc_equivalent": "304",
        },
    ]

    sections = []
    for s in CRITICAL_BNSS:
        s["source"]     = "BNSS 2023"
        s["act"]        = "Bharatiya Nagarik Suraksha Sanhita 2023"
        s["chunk_type"] = "statute"
        sections.append(s)

    print(f"Created {len(sections)} hardcoded critical BNSS sections")
    return sections


def main():
    ap = argparse.ArgumentParser(description="Ingest BNSS 2023 into NyayaMitra index")
    ap.add_argument("--pdf",       type=str, default=None,
                    help="Path to BNSS 2023 PDF file")
    ap.add_argument("--text",      type=str, default=None,
                    help="Path to BNSS 2023 plain text file")
    ap.add_argument("--index_dir", type=str, default="indexes",
                    help="Path to indexes/ directory")
    ap.add_argument("--mode",      type=str, default="append",
                    choices=["append", "rebuild"],
                    help="append = add to existing index (default)")
    ap.add_argument("--hardcoded", action="store_true",
                    help="Use hardcoded critical sections (no PDF needed)")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)

    if args.hardcoded or (not args.pdf and not args.text):
        print("Using hardcoded critical BNSS sections...")
        print("(Pass --pdf or --text for full BNSS ingestion)")
        sections = create_bnss_from_known_sections()
    elif args.pdf:
        sections = parse_bnss_from_pdf(args.pdf)
    else:
        text = Path(args.text).read_text(encoding="utf-8")
        sections = parse_bnss_from_text(text)

    embed_and_append_to_index(sections, index_dir)

    print("\nTo verify BNSS is in the index, run:")
    print("  python scripts/05_test_retrieval.py "
          "--query 'bail conditions BNSS 2023 Section 480'")


if __name__ == "__main__":
    main()
