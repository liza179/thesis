import os
from glob import glob
from dotenv import load_dotenv
from anthropic import Anthropic
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def ask_llm(prompt, max_tokens=15000, model="claude-haiku-4-5"):
    result =  client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="claude-haiku-4-5",
    )
    return result.content[0].text

def translate_chunk(text):
    PROMPT = """
### TASK ###
You will be given a text in Spanish, from Central Bank of Argentina, its monthly report.
It is parsed from a PDF file, some words might be splitted or parsed incorrectly.
Your task is to translate the text to English, keeping as close to the text as possible.
DO NOT SKIP ANY SECTION OF THE TEXT.

### INPUT ###
""".strip()
    prompt = f"{PROMPT}\n\n{text}"
    result = ask_llm(prompt)
    return result.split("###")[-1].split("TO ENGLISH")[-1].split("TRANSLATION")[-1].strip()

def translate(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=0)
    chunks = splitter.split_text(text)
    for chunk in chunks:
        yield translate_chunk(chunk)


def translate_all():
    mapping = []
    for input_file in glob("parsed/bol*.txt"):
        digits = os.path.basename(input_file).split(".")[0].split("bol")[1]
        month = digits[:2]
        year = digits[2:]
        output_file = f"translated/{year}_{month}.md"
        mapping.append((input_file, output_file))
    mapping.sort(key=lambda x: x[1])
    for input_file, output_file in mapping:
        logger.info(f"Translating {input_file} to {output_file}")
        if output_file in glob("translated/*.md"):
            logger.info(f"Skipping {output_file}")
            continue
        try:
            with open(input_file, "r") as f:
                text = f.read()
            with open(output_file, "w") as f:
                for chunk in translate(text):
                    f.write(chunk + "\n")
                    f.flush()
                    logger.debug(f"Translated chunk of size {len(chunk)}")
            logger.success(f"Translated {input_file} to {output_file}")
        except Exception as e:
            logger.error(f"Error translating {input_file}: {e}")

if __name__ == "__main__":
    translate_all()