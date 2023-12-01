from typing import List

from indexify_extractor_sdk import EmbeddingSchema, ExtractorSchema
from indexify_extractor_sdk.base_embedding import (
    BaseEmbeddingExtractor,
    EmbeddingInputParams,
)

from openai import OpenAI


class OpenAIEmbeddingExtractor(BaseEmbeddingExtractor):
    def __init__(self):
        super(OpenAIEmbeddingExtractor, self).__init__(max_context_length=128)
        self.model_name = "text-embedding-ada-002"
        self.client = OpenAI()

    def extract_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.embeddings.create(input=texts, model=self.model_name)
        return [data.embedding for data in embeddings.data]

    def extract_query_embeddings(self, query: str) -> List[float]:
        return (
            self.client.embeddings.create(input=query, model=self.model_name)
            .data[0]
            .embedding
        )

    def schemas(self) -> ExtractorSchema:
        input_params = EmbeddingInputParams()
        return ExtractorSchema(
            input_params=input_params.model_dump_json(),
            output_schemas={
                "embedding": EmbeddingSchema(distance_metric="cosine", dim=1536)
            },
        )
