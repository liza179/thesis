import re
from pypdf import PdfReader
import glob
import os
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text(extraction_mode="layout")
    return text

def remove_consecutive_whitespace(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def main():
    pdf_files = glob.glob("monthly_reports/*.pdf")
    for pdf_path in tqdm(pdf_files):
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        text = extract_text_from_pdf(pdf_path)
        text = remove_consecutive_whitespace(text)
        out_path = f"parsed/{filename}.txt"
        with open(out_path, "w") as f:
            f.write(text)

if __name__ == "__main__":
    main()