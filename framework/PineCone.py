import os

import pinecone

from framework.VectorOps import VectorOps
api_key = os.getenv('api_key')

class PineCone(VectorOps):
    def __init__(self):
        super().__init__()
        self.vectordb = 'Pinecone'
        pinecone.init(
            api_key=api_key,
            environment="us-east-1-aws"
        )

    # noinspection PyMethodMayBeStatic
    def upsert_embeds(self, index, vectors):
        index = pinecone.Index(index)
        index.upsert(vectors=vectors)

    def create_index(self, **kwargs):
        for key, value in kwargs.items():
            if key == "index_name":
                index_name = value
            if key == "dimension":
                dimension = value
            if key == "metric":
                metric = value
            if key == "metadata_config":
                metadata_config = value

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
            index_name,
            dimension=dimension,
            metric=metric,
            metadata_config=metadata_config
        )
