import logging


class FrameworkFilter(logging.Filter):
    """
    Filters out and suppresses clutter/framework-level diagnostic logs
    unless log level is DEBUG.
    """

    def __init__(self, name=""):
        super().__init__(name)
        self.suppressed_keywords = [
            "processing event",
            "event ::",
            "contextupdate",
            "listen",
            "botintent",
            "userintent",
            "startinternalsystemaction",
            "internalsystemactionfinished",
            "action dispatcher internals",
            "action registration",
            "registered actions",
            "http request",
        ]
        self.suppressed_loggers = [
            "nemoguardrails",
            "pinecone",
            "urllib3",
            "httpx",
            "google.genai",
            "langchain",
            "openai",
            "google",
            "gcs",
        ]

    def filter(self, record):
        # Normalize application logger names to 'SupportSphere'
        if record.name.startswith("src.") or record.name.startswith("vectordb_ingestion") or record.name == "__main__":
            record.name = "SupportSphere"

        # Always allow DEBUG level logs to pass to target handlers
        if record.levelno <= logging.DEBUG:
            return True

        # Downgrade framework loggers
        logger_name = record.name.lower()
        if any(sl in logger_name for sl in self.suppressed_loggers):
            return False

        # Downgrade clutter keywords
        msg = str(record.msg).lower()
        if any(kw in msg for kw in self.suppressed_keywords):
            return False

        return True
