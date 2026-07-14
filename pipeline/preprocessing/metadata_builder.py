from pathlib import Path
from urllib.parse import urlparse


class MetadataBuilder:
    """
    Builds standardized metadata for every support article.
    """

    def build(self, article: dict, file_path: str) -> dict:
        """
        Build metadata from a parsed markdown article.
        """

        company = self._get_company(file_path)

        article_id = article.get("article_id", "")

        title = article.get("title", "").split("\n")[0].strip()

        title_slug = article.get("title_slug", "")

        breadcrumbs = article.get("breadcrumbs", [])

        product_area = self._get_product_area(breadcrumbs, file_path, company)

        url = article.get("source_url", "")

        return {
            "doc_id": f"{company}_{article_id}",

            "article_id": article_id,

            "company": company,

            "product_area": product_area,

            "breadcrumbs": breadcrumbs,

            "title": title,

            "title_slug": title_slug,

            "url": url,

            "domain": self._get_domain(url),

            "last_updated": article.get("last_updated", ""),

            "file_name": article.get("file_name", ""),
            "source_type": "documentation",
        }

    def _get_company(self, file_path: str) -> str:
        """
        Determine company from folder name.
        """

        path = Path(file_path).parts
        path = [part.lower() for part in path]

        if "hackerrank" in path:
            return "hackerrank"

        if "claude" in path:
            return "claude"

        if "visa" in path:
            return "visa"

        return "unknown"

    def _get_product_area(self, breadcrumbs: list, file_path: str, company: str) -> str:
        """
        Product area comes from the first breadcrumb.
        Falls back to the immediate parent folder of the file when breadcrumbs
        are missing (e.g. Visa pages that have no breadcrumb metadata).
        """
        path_str = file_path.replace("\\", "/").lower()
        if "hackerrank_community" in path_str:
            return "community"

        if "travelers-cheques" in path_str:
            return "travel_support"

        if "travel-support" in path_str:
            return "travel_support"

        if path_str.endswith("support.md") or path_str.endswith("support.html"):
            return "general_support"

        if breadcrumbs:
            return breadcrumbs[0].strip().lower().replace(" ", "_")

        # Fallback: derive from file path parent folder
        path = Path(file_path)
        parent = path.parent.name.lower()

        # If the parent folder IS the company name, go one level up
        if parent == company:
            parent = path.parent.parent.name.lower()

        return parent.replace("-", "_").replace(" ", "_")

    def _get_domain(self, url: str) -> str:
        """
        Extract domain name from source URL.
        """

        if not url:
            return ""

        return urlparse(url).netloc
