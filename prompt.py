import os

import openai
import pinecone

embed_model = "text-embedding-ada-002"
index_name = 'wiki01'
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv('api_key')
pinecone.init(
    api_key=api_key,
    environment="us-east-1-aws"
)
# connect to index
index = pinecone.Index(index_name)
limit = 3750
openai.api_key = os.getenv('API_TOKEN')


def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    print("--res")
    print(res)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()

if __name__=="__main__":
    query = (
            "Where is Munich Kouros located?"
    )
    query_with_contexts = retrieve(query)
    print(query_with_contexts)
    answer = complete(query_with_contexts)
    print(answer)