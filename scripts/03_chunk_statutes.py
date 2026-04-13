"""
03_chunk_statutes.py  —  WORKING VERSION
Uses confirmed working PDF sources. Auto-downloads them.

Install:  pip install pymupdf requests tqdm
Run:      python 03_chunk_statutes.py
"""

import re, os, json, argparse, requests
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("data/processed/chunks")
RAW_DIR    = Path("data/raw/statutes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORDS = 400
OVERLAP   = 60

SOURCES = {
    "ipc": {
        "url"     : "https://www.ncib.in/pdf/indian-penal-code.pdf",
        "filename": "ipc_1860.pdf",
    },
    "bns": {
        "url"     : "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf",
        "filename": "bns_2023.pdf",
    },
}

def download_pdf(url, dest):
    if dest.exists():
        print(f"  Already exists: {dest}"); return True
    print(f"  Downloading {dest.name}...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
        r = requests.get(url, headers=headers, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk); bar.update(len(chunk))
        print(f"  ✅ {dest}  ({dest.stat().st_size//1024} KB)")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}"); return False

def pdf_to_text(path):
    import fitz
    doc = fitz.open(str(path))
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    return text

def parse_ipc(text):
    # IPC format: "Section 302. Punishment for murder"
    pattern = re.compile(r'Section\s+(\d{1,4}[A-Za-z]?)\s*[.\-]\s*([^\n]{3,120})', re.IGNORECASE)
    matches = list(pattern.finditer(text))
    print(f"  Found {len(matches)} IPC sections")
    sections = []
    for i, m in enumerate(matches):
        num   = m.group(1).strip()
        title = m.group(2).strip().rstrip('.')
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        body  = text[m.start():end].strip()
        if len(body.split()) >= 5:
            sections.append({"section_num": num, "section_title": title,
                              "text": f"IPC Section {num} — {title}\n\n{body}",
                              "source": "IPC_1860", "description": "Indian Penal Code, 1860"})
    return sections

def parse_bns(text):
    # BNS format: bare "101. ..." at line start
    pattern = re.compile(r'(?:^|\n)\s*(\d{1,3}[A-Z]?)\.\s+', re.MULTILINE)
    all_m = list(pattern.finditer(text))
    # Keep only sections 1-358
    matches = [m for m in all_m if 1 <= int(re.sub(r'[A-Z]','', m.group(1)) or 0) <= 360]
    print(f"  Found {len(matches)} BNS sections")
    sections = []
    for i, m in enumerate(matches):
        num   = m.group(1).strip()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        body  = text[m.start():end].strip()
        title = body.split('\n')[0][:80].strip().rstrip('.')
        if len(body.split()) >= 5:
            sections.append({"section_num": num, "section_title": title,
                              "text": f"BNS 2023 Section {num} — {title}\n\n{body}",
                              "source": "BNS_2023", "description": "Bharatiya Nyaya Sanhita, 2023"})
    return sections

def sub_chunk(sec, max_w, overlap):
    words = sec["text"].split()
    if len(words) <= max_w:
        return [{**sec, "sub_chunk_idx": 0, "total_sub_chunks": 1}]
    chunks, start, idx = [], 0, 0
    while start < len(words):
        end = min(start + max_w, len(words))
        c = {**sec, "text": " ".join(words[start:end]), "sub_chunk_idx": idx}
        chunks.append(c)
        if end == len(words): break
        start += max_w - overlap; idx += 1
    for c in chunks: c["total_sub_chunks"] = len(chunks)
    return chunks

# ── 32 key built-in sections (IPC + BNS) — always included ──────────────────
BUILTIN = [
    ("IPC_1860","120B","Criminal Conspiracy","Whoever is a party to a criminal conspiracy to commit an offence punishable with death, imprisonment for life or rigorous imprisonment for a term of two years or upwards shall be punished in the same manner as if he had abetted such offence. [IPC §120B]"),
    ("IPC_1860","299","Culpable Homicide","Whoever causes death by doing an act with intention of causing death, or causing such bodily injury as is likely to cause death, or with knowledge that he is likely to cause death, commits culpable homicide. [IPC §299]"),
    ("IPC_1860","300","Murder — Definition","Culpable homicide is murder if the act is done with intention of causing death, or causing such bodily injury as the offender knows likely to cause death. Exceptions: grave provocation, exceeded right of private defence, consent, necessity. [IPC §300]"),
    ("IPC_1860","302","Murder — Punishment","Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine. [IPC §302]"),
    ("IPC_1860","304","Culpable Homicide Not Amounting to Murder","Whoever commits culpable homicide not amounting to murder shall be punished with imprisonment for life, or imprisonment up to ten years, and fine. [IPC §304]"),
    ("IPC_1860","304B","Dowry Death","Death of woman caused by burns/bodily injury within 7 years of marriage with prior dowry cruelty: dowry death. Punishment: not less than 7 years, up to life imprisonment. [IPC §304B]"),
    ("IPC_1860","307","Attempt to Murder","Whoever does any act with intention/knowledge that if death were caused he would be guilty of murder — imprisonment up to 10 years and fine; if hurt caused, imprisonment up to life. [IPC §307]"),
    ("IPC_1860","354","Assault on Woman","Whoever assaults or uses criminal force to any woman intending to outrage her modesty — imprisonment not less than 1 year, extendable to 5 years, and fine. [IPC §354]"),
    ("IPC_1860","375","Rape — Definition","A man commits rape if he penetrates or inserts against will, without consent, consent by force/fraud/intoxication, or when woman is below 18. [IPC §375]"),
    ("IPC_1860","376","Rape — Punishment","Rigorous imprisonment not less than 7 years extendable to life, and fine. Gang rape: not less than 20 years to life. Rape on woman below 12: not less than 20 years to life or death. [IPC §376]"),
    ("IPC_1860","378","Theft — Definition","Whoever intending to take dishonestly any moveable property out of another's possession without consent moves that property commits theft. [IPC §378]"),
    ("IPC_1860","379","Theft — Punishment","Whoever commits theft shall be punished with imprisonment up to 3 years, or fine, or both. [IPC §379]"),
    ("IPC_1860","390","Robbery","Theft is robbery if the offender voluntarily causes/attempts to cause death, hurt, or wrongful restraint in the course of committing theft. Extortion is robbery if the offender puts person in fear of instant death/hurt/wrongful restraint. [IPC §390]"),
    ("IPC_1860","392","Robbery — Punishment","Rigorous imprisonment up to 10 years and fine. On highway between sunset and sunrise: up to 14 years. [IPC §392]"),
    ("IPC_1860","395","Dacoity — Punishment","Imprisonment for life, or rigorous imprisonment up to 10 years, and fine. [IPC §395]"),
    ("IPC_1860","415","Cheating — Definition","Whoever by deceiving any person dishonestly induces delivery of property, or consents to retain property, or does/omits to do anything he would not otherwise do, commits cheating. [IPC §415]"),
    ("IPC_1860","420","Cheating — Punishment","Whoever cheats and induces delivery of property or destruction of valuable security — imprisonment up to 7 years and fine. [IPC §420]"),
    ("IPC_1860","498A","Cruelty by Husband","Whoever being the husband or relative of the husband subjects a woman to cruelty shall be punished with imprisonment up to 3 years and fine. Cruelty includes: conduct likely to drive to suicide, grave injury to life/limb/health, or harassment for dowry. [IPC §498A]"),
    ("IPC_1860","504","Intentional Insult","Whoever intentionally insults thereby provoking breach of peace or commission of an offence — imprisonment up to 2 years, or fine, or both. [IPC §504]"),
    ("IPC_1860","506","Criminal Intimidation","Whoever commits criminal intimidation — up to 2 years or fine or both. If threat is of death/grievous hurt/destruction by fire/offence punishable with death or life imprisonment — up to 7 years, or fine, or both. [IPC §506]"),
    # BNS 2023
    ("BNS_2023","63","Rape — Definition (BNS)","A man commits rape under BNS if he penetrates or inserts against will, without consent, consent by force/fraud/intoxication, or when woman is below 18 years. [BNS 2023 §63 — replaces IPC §375]"),
    ("BNS_2023","64","Rape — Punishment (BNS)","Rigorous imprisonment not less than 10 years extendable to life and fine. Gang rape by 5+ persons: not less than 20 years to life. [BNS 2023 §64 — replaces IPC §376]"),
    ("BNS_2023","101","Murder (BNS)","Whoever commits murder shall be punished with death or imprisonment for life and fine, or in exceptional cases up to 7 years and fine. [BNS 2023 §101 — replaces IPC §302]"),
    ("BNS_2023","103","Culpable Homicide (BNS)","Whoever commits culpable homicide not amounting to murder shall be punished with life imprisonment, or imprisonment up to 10 years, and fine. [BNS 2023 §103 — replaces IPC §304]"),
    ("BNS_2023","106","Causing Death by Negligence (BNS)","Rash or negligent act causing death not amounting to culpable homicide — imprisonment up to 5 years and fine. Registered medical practitioner — up to 2 years and fine. [BNS 2023 §106 — replaces IPC §304A]"),
    ("BNS_2023","109","Attempt to Murder (BNS)","Whoever does any act with intention/knowledge that if death were caused he would be guilty of murder — imprisonment up to 10 years and fine. [BNS 2023 §109 — replaces IPC §307]"),
    ("BNS_2023","111","Organised Crime (BNS) — NEW","Member of organised crime syndicate committing continuing unlawful activity (kidnapping, extortion, trafficking, financial crime, cybercrime): death or life imprisonment if activity results in death; otherwise rigorous imprisonment not less than 5 years. [BNS 2023 §111 — new section]"),
    ("BNS_2023","113","Terrorist Act (BNS) — NEW","Whoever commits a terrorist act, or prepares/participates/facilitates/organises/trains for such act — death or imprisonment for life. [BNS 2023 §113 — new section]"),
    ("BNS_2023","303","Theft (BNS)","Whoever commits theft — imprisonment up to 3 years, or fine, or both. [BNS 2023 §303 — replaces IPC §379]"),
    ("BNS_2023","309","Dacoity (BNS)","Whoever commits dacoity — imprisonment for life, or rigorous imprisonment up to 10 years, and fine. [BNS 2023 §309 — replaces IPC §395]"),
    ("BNS_2023","316","Cruelty by Husband (BNS)","Whoever being the husband or his relative subjects a woman to cruelty — imprisonment up to 3 years and fine. [BNS 2023 §316 — replaces IPC §498A]"),
    ("BNS_2023","318","Cheating (BNS)","Whoever cheats — up to 3 years or fine or both. Cheating inducing delivery of property — up to 7 years and fine. [BNS 2023 §318 — replaces IPC §420]"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc_pdf",      default=None)
    parser.add_argument("--bns_pdf",      default=None)
    parser.add_argument("--builtin_only", action="store_true")
    parser.add_argument("--output",       default="data/processed/chunks")
    args = parser.parse_args()

    OUTPUT = Path(args.output)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    cid = 0

    if not args.builtin_only:
        jobs = []
        # IPC
        ipc_path = Path(args.ipc_pdf) if args.ipc_pdf else RAW_DIR / SOURCES["ipc"]["filename"]
        if not args.ipc_pdf:
            download_pdf(SOURCES["ipc"]["url"], ipc_path)
        if ipc_path.exists():
            jobs.append((ipc_path, "ipc"))
        # BNS
        bns_path = Path(args.bns_pdf) if args.bns_pdf else RAW_DIR / SOURCES["bns"]["filename"]
        if not args.bns_pdf:
            download_pdf(SOURCES["bns"]["url"], bns_path)
        if bns_path.exists():
            jobs.append((bns_path, "bns"))

        for pdf_path, kind in jobs:
            print(f"\nParsing {pdf_path.name}...")
            text     = pdf_to_text(pdf_path)
            sections = parse_ipc(text) if kind == "ipc" else parse_bns(text)
            for sec in tqdm(sections, desc=f"Chunking"):
                for chunk in sub_chunk(sec, MAX_WORDS, OVERLAP):
                    all_chunks.append({"chunk_id": f"statute_{cid}", "chunk_type": "statute",
                                       **{k: chunk[k] for k in ["text","section_num","section_title","source","description"]},
                                       "sub_chunk_idx": chunk.get("sub_chunk_idx",0),
                                       "total_sub_chunks": chunk.get("total_sub_chunks",1)})
                    cid += 1

    # Always add built-ins that aren't already present
    existing = {(c["source"], c["section_num"]) for c in all_chunks}
    for source, num, title, text in BUILTIN:
        if (source, num) in existing: continue
        all_chunks.append({"chunk_id": f"statute_{cid}", "chunk_type": "statute",
                            "text": text, "section_num": num, "section_title": title,
                            "source": source,
                            "description": "Indian Penal Code, 1860" if source=="IPC_1860" else "Bharatiya Nyaya Sanhita, 2023",
                            "sub_chunk_idx": 0, "total_sub_chunks": 1})
        cid += 1

    out_path = OUTPUT / "statute_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter
    by_src = Counter(c["source"] for c in all_chunks)
    print(f"\n✅ Total statute chunks: {len(all_chunks)}")
    for s, n in by_src.items(): print(f"  {s}: {n}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()