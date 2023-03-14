import os
from time import sleep

import openai
from tqdm.auto import tqdm

from framework.PineCone import PineCone

openai.api_key = os.getenv('API_TOKEN')
embed_model = "text-embedding-ada-002"


class Embeddings():
    def __init__(self, vectorDb):
        self.module = "Embeddings"
        self.vectorDb = vectorDb
        if self.vectorDb == "Pinecone":
            self.embedstore = PineCone()

    def create_index_for_embeds(self, **kwargs):
        self.embedstore.create_index(**kwargs)

    def create_embeddings(self, emb_data, batch_size=100):
        print('--step 2 entered---')
        print(len(emb_data))
        for i in tqdm(range(0, len(emb_data), batch_size)):
            i_end = min(len(emb_data), i + batch_size)
            meta_batch = emb_data[i:i_end]
            ids_batch = [x['id'] for x in meta_batch]
            texts = [x['text'][1:1000] for x in meta_batch]
            print(len(texts))
            try:
                print('i am here')
                res = openai.Embedding.create(input=texts, engine=embed_model)
                print('--res--')
                print(res)
            except:
                done = False
                while not done:
                    sleep(5)
                    try:
                        print('I am here too..')
                        res = openai.Embedding.create(input=texts, engine=embed_model)
                        done = True
                    except:
                        pass
            embeds = [record['embedding'] for record in res['data']]
            meta_batch = [{
                'title': x['title'],
                'text' : x['text'][1:1000]
            } for x in meta_batch]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            return to_upsert

    # noinspection PyMethodMayBeStatic
    def upsert_embeds(self, index, vectors):
        self.embedstore .upsert_embeds(index,vectors)

