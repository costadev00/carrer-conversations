from pathlib import Path
from PyPDF2 import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent / "me"
SOURCES = ["linkedin", "lattes"]

for name in SOURCES:
    pdf_path = BASE_DIR / f"{name}.pdf"
    txt_path = BASE_DIR / f"{name}.txt"
    if not pdf_path.exists():
        print(f"Skipping {pdf_path} (missing)")
        continue
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.append(text)
    txt_path.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Wrote {txt_path} ({len(chunks)} pages)")
