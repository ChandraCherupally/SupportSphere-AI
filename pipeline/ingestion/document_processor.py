from pathlib import Path
# pyrefly: ignore [missing-import]
from pipeline.preprocessing.markdown_parser import MarkdownParser
# pyrefly: ignore [missing-import]
from pipeline.preprocessing.metadata_builder import MetadataBuilder
# pyrefly: ignore [missing-import]
from pipeline.preprocessing.article_enricher import ArticleEnricher
# pyrefly: ignore [missing-import]
from pipeline.preprocessing.chunker import Chunker


class DocumentProcessor:
    """
    Processes markdown documents into chunk objects.
    """


    def __init__(self):
        self.parser = MarkdownParser()
        self.metadata_builder = MetadataBuilder()
        self.article_enricher = ArticleEnricher()
        self.chunker = Chunker()


    def process_document(self, file_path: str):
        article = self.parser.parse(file_path)
        metadata = self.metadata_builder.build(article, file_path)
        article = self.article_enricher.enrich(article,company=metadata["company"])
        chunks = self.chunker.chunk(article, metadata)

        return chunks


    def process_directory(self, directory: str):

        all_chunks = []
        directory = Path(directory)
        markdown_files = directory.rglob("*.md")

        for file_path in markdown_files:
            chunks = self.process_document(str(file_path))
            all_chunks.extend(chunks)
        
        return all_chunks
        



