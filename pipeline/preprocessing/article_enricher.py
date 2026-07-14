import re


class ArticleEnricher:
    """
    Extracts additional structured information
    from a support article.
    """

    def enrich(self, article: dict,company: str) -> dict:
        """
        Enrich article with additional metadata.
        """

        content = article["content"]
        article["related_articles"] = self._extract_related_articles(content, company)
        article["support_links"] = self._extract_support_links(content,company)
        article["external_links"] = self._extract_external_links(content)

        return article


    def _extract_related_articles(self, text: str, company: str):

        related_articles = []

        pattern = (r"Related Articles\s*(.*?)(?:\n#|\Z)")

        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

        if not match:
            return related_articles

        section = match.group(1)

        links = re.findall(r"\[(.*?)\]\((.*?)\)", section)
        
        for title, url in links:

            article_id = ""
            doc_id = ""
            id_match = re.search(r"/articles/(\d+)", url)
        

            if id_match:
                article_id = id_match.group(1)
                doc_id = f"{company}_{article_id}"

            related_articles.append({"title": title.strip(), 
                                     "article_id": article_id, 
                                     "doc_id": doc_id,
                                     "url": url})

        return related_articles


    def _extract_support_links(self, text: str,company: str):

        links = []                
        
        matches = re.findall(r"\[(.*?)\]\((/articles/.*?)\)", text)

        for title, url in matches:

            article_id = ""
            doc_id = ""

            id_match = re.search(r"/articles/(\d+)", url)

            if id_match:
                article_id = id_match.group(1)
                doc_id = f"{company}_{article_id}"


            links.append({"title": title.strip(),"article_id": article_id, "doc_id": doc_id, "url": url})

        return links


    def _extract_external_links(self, text: str):

        links = []

        matches = re.findall(r"\[(.*?)\]\((https?://.*?)\)", text)

        for title, url in matches:
            links.append({"title": title.strip(), "url": url})

        return links