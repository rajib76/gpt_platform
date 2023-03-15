import os

import openai
import pinecone


class PromptEngine():
    def __init__(self, index_name):
        openai.api_key = os.getenv('API_TOKEN')
        api_key = os.getenv('api_key')
        pinecone.init(
            api_key=api_key,
            environment="us-east-1-aws"
        )
        self.index = pinecone.Index(index_name)
        self.embed_model = "text-embedding-ada-002"
        self.engine = "text-davinci-003"
        self.limit = 3750

    def retrieve(self, query):
        res = openai.Embedding.create(
            input=[query],
            engine=self.embed_model
        )

        # retrieve from Pinecone
        xq = res['data'][0]['embedding']

        # get relevant contexts
        res = self.index.query(xq, top_k=3, include_metadata=True)
        contexts = [
            x['metadata']['text'] for x in res['matches']
        ]

        # build our prompt with the retrieved contexts included
        prompt_start = (
                "Answer the question based on the context below.\n\n" +
                "Context:\n"
        )
        prompt_end = (
            f"\n\nQuestion: {query}\nAnswer:"
        )
        # append contexts until hitting limit
        for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= self.limit:
                prompt = (
                        prompt_start +
                        "\n\n---\n\n".join(contexts[:i - 1]) +
                        prompt_end
                )
                break
            elif i == len(contexts) - 1:
                prompt = (
                        prompt_start +
                        "\n\n---\n\n".join(contexts) +
                        prompt_end
                )
        return prompt

    def complete(self, prompt):
        # query text-davinci-003
        res = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=0,
            max_tokens=400,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return res['choices'][0]['text'].strip()


if __name__ == "__main__":
    query = (
        "Where is Munich Kouros located?"
    )
    pe = PromptEngine('wiki01')
    query_with_contexts = pe.retrieve(query)
    answer = pe.complete(query_with_contexts)
    print(answer)
