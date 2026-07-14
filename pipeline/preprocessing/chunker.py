import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from src.config import MAX_SECTION_SIZE, SECTION_OVERLAP
import hashlib

class Chunker:
    """
    Splits a markdown article into semantic chunks.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_SECTION_SIZE, 
                                                            chunk_overlap=SECTION_OVERLAP)


    def chunk(self, article: dict, metadata: dict) -> list:
        """
        Split article into semantic chunks.
        """

        sections = self._split_sections(article["content"])
        chunks = []
        chunk_number = 1

        for section in sections:
            heading = section["heading"]
            content = section["content"]
            
            content = content.strip()
            if not content:
                continue
            if heading.lower() == "related articles":
                continue

            if len(content) <= MAX_SECTION_SIZE:
                chunks.append(self._build_chunk(chunk_number, heading, content, metadata, article))
                chunk_number += 1

            else:
                split_chunks = self.text_splitter.split_text(content)

                for chunk in split_chunks:
                    chunks.append(self._build_chunk(chunk_number,heading,chunk,metadata, article))
                    chunk_number += 1
        
        return chunks


    def _split_sections(self, text: str):
        # Split only on H1 and H2 headings — H3 sub-sections stay merged
        # into their parent section so small sub-entries (e.g. individual
        # bank contacts) keep enough context for retrieval.
        pattern = r"(?=^#{1,2}\s)"
        raw_sections = re.split(pattern, text, flags=re.MULTILINE)
        sections = []

        for section in raw_sections:
            section = section.strip()
            if not section:
                continue

            lines = section.splitlines()
            heading = ""
            content = section

            if lines[0].startswith("#"):
                heading = lines[0].lstrip("#").strip()
                content = "\n".join(lines[1:]).strip()
            
            if heading.lower() == "frequently asked questions":
                faq_sections = re.split(r"(?=^####\s)", content,flags=re.MULTILINE)

                for faq in faq_sections:
                    faq = faq.strip()
                    if not faq:
                        continue

                    faq_lines = faq.splitlines()
                    faq_heading = faq_lines[0].replace("####", "").strip()
                    faq_content = "\n".join(faq_lines[1:]).strip()
                    sections.append({"heading": faq_heading, "content": faq_content})
                continue    

            sections.append({"heading": heading,"content": content})

        return sections

    

    def _build_chunk(self,chunk_number,heading,content,metadata, article):

        chunk_text = f"{heading}\n\n{content}" if heading else content
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        return {"id": f'{metadata["doc_id"]}_{chunk_number:03}',
                "text": chunk_text,
                "metadata": {
                            
                            **metadata,
                            "content_hash": content_hash,
                            "section": heading,
                            "section_path": [*metadata["breadcrumbs"],heading],
                            "chunk_index": chunk_number,
                            "chunk_type": self._get_chunk_type(heading, content),
                            "word_count": len(content.split()),
                            "char_count": len(content),
                            "token_estimate": round(len(chunk_text) / 4),
                            "related_doc_ids": [article["doc_id"]for article in article["related_articles"]]
                            }
        }
    







    def _get_chunk_type(self, heading, content):

        heading = heading.lower()

        if heading.lower() in ["frequently asked questions", "troubleshooting"]:
            return "troubleshooting"

        if "troubleshooting" in heading:
            return "troubleshooting"

        if "faq" in heading:
            return "faq"

        if "overview" in heading:
            return "overview"

        if "benefit" in heading:
            return "concept"

        if "prerequisite" in heading:
            return "prerequisite"

        if "note" in heading:
            return "note"

        # Detect numbered procedures
        if re.search(r"^\s*1\.", content, re.MULTILINE):
            return "procedure"

        # Detect common procedural headings
        procedure_keywords = [
            "how",
            "getting started",
            "setup",
            "install",
            "configure",
            "configuration",
            "create",
            "delete",
            "reset",
            "enable",
            "disable",
            "manage",
            "update"
        ]

        if any(keyword in heading for keyword in procedure_keywords):
            return "procedure"

        return "general"


