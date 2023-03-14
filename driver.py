import ast

from engine.Embeddings import Embeddings


def get_data_for_embedding():
    mod_data = []
    with open("./data/blog.json", 'r') as f:
        data = ast.literal_eval(f.read())
        for item in data:
            mod_data.append(item)

    return mod_data


if __name__ == "__main__":
    vectorDb = "Pinecone"
    emb = Embeddings(vectorDb)
    index_name = 'wiki01'
    # Create the index to store the embeddings
    emb.create_index_for_embeds(index_name=index_name,
                                dimension=1536,
                                metric='cosine',
                                metadata_config={'indexed': ['title']})

    # new_data = get_data_to_be_embedded()
    print('--step 1---')
    new_data = get_data_for_embedding()
    print('--step 2---')
    vectors = emb.create_embeddings(new_data)
    print('--step 3---')
    emb.upsert_embeds(index_name, vectors)
