from typing import Iterable, List

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from settings import settings


class SemanticDistanceCalculator:
    """
    A utility class for calculating semantic similarity between a query and a list of sentences
    using embeddings from an OpenAI-compatible API.
    """

    def __init__(self, model: str | None = None, api_base: str | None = None):
        """
        Initializes the SemanticDistanceCalculator with OpenAI-compatible embeddings.
        """
        self.model = model or settings.openai_emb_model
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=api_base or settings.openai_emb_api_base,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    def get_top_k_sentences(self, query, sentences, k):
        """
        Returns the top-k sentences most semantically similar to the query.
        """
        if not sentences:
            return []

        embeddings = self._embed_texts([query, *sentences])
        similarity_scores = cosine_similarity([embeddings[0]], embeddings[1:])

        similarity_scores = list(similarity_scores[0])

        sorted_sentences = [
            x
            for _, x in sorted(
                zip(similarity_scores, sentences),
                key=lambda pair: pair[0],
                reverse=True,
            )
        ]

        return sorted_sentences[:k]
