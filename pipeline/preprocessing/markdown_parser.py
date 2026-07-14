from pathlib import Path
import frontmatter
import re

class MarkdownParser:
    """
    Parses a markdown support article and extracts
    the frontmatter metadata and markdown body.
    """

    def parse(self, file_path: str) -> dict:
        """
        Parse a markdown file.
        Returns: {"title": ..., "title_slug": ..., "article_id": ..., "source_url": ..., "last_updated": ..., "breadcrumbs": [...], "content": ...}
        """
        post = frontmatter.load(file_path)

        article_id = post.get("article_id", "")
        

        if not article_id:
            article_slug = post.get("article_slug", "")
            if article_slug:
                article_id = article_slug.split("-")[0]

        match = re.search(r"^#\s+(.+)$", post.content, re.MULTILINE)
        
        if match:
             title = match.group(1).strip()
        else:
             title = post.get("title", "")

        return {
            "title": title,
            "title_slug": post.get("title_slug", ""),
            "article_id": article_id,            
            "source_url": post.get("source_url", ""),
            "last_updated": (post.get("last_updated_iso")
                            or post.get("last_updated_exact")
                            or post.get("last_modified")
                            or ""),
            "breadcrumbs": post.get("breadcrumbs", []),
            "content" : self._clean_markdown(post.content),
            "file_name": Path(file_path).name,
        }
    
    def _clean_markdown(self, text: str):

        # Remove markdown images
        text = re.sub(r'!\[.*?\]\(.*?\)','',text)

        # Remove HTML img tags
        text = re.sub(r'<img[^>]*>','',text)

        # Remove escaped backslashes
        text = re.sub(r'\\','',text)

        # Remove extra blank lines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove "Last Updated"
        text = re.sub(r"_Last updated:.*?_","",text,flags=re.DOTALL)

        # Remove trailing spaces
        text = re.sub(r"[ \t]+\n","\n",text)

        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}","\n\n",text)

        return text.strip()
    

"""
if __name__ == "__main__":
    md = MarkdownParser()
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    print(PROJECT_ROOT)
    path = PROJECT_ROOT / "data/claude/amazon-bedrock/10280779-how-do-i-learn-more-about-amazon-and-anthropic-s-strategic-collaboration.md"
    print(path)
    result = md.parse(path) 
    #print(result['content'])
"""






















"""
Why use python-frontmatter instead of parsing manually?

Without this library, you would have to:

Open the file.
Detect the --- delimiters.
Parse the YAML section.
Separate metadata from the Markdown body.
Handle malformed or missing front matter.

frontmatter.load(file_path) performs all of these steps in a single call, returning a convenient object that exposes metadata through post.get(...) (or post[...]) and the Markdown body through post.content. This makes your parser much simpler, cleaner, and less error-prone.
"""    


