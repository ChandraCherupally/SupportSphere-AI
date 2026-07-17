from pydantic import BaseModel


class RerankerResponse(BaseModel):
    """
    Structured response returned by Gemini.

    Example
    -------
    {
        "ranking": [
            "chunk_12",
            "chunk_5",
            "chunk_8"
        ]
    }
    """

    ranking: list[str]