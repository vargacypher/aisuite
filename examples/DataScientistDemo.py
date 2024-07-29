from dotenv import load_dotenv

load_dotenv()
import requests
from sentence_transformers import SentenceTransformer
import chromadb

from aimodels.client import MultiFMClient


url = "https://gutenberg.net.au/ebooks02/0200041.txt"
response = requests.get(url)
data = response.text.replace("\n", " ")
docs = [data[idx * 1000 : (idx + 1) * 1000] for idx in range(len(data) // 1000)]
docs.append(data[(len(data) // 1000) * 1000 :])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
data_emb = model.encode(docs)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="SampleDB")
collection.add(
    embeddings=data_emb.tolist(),
    documents=docs,
    ids=[str(idx) for idx in range(len(data_emb))],
)  # Doc ID's are requried

question = "What is happening with cars?"
question_emb = model.encode(question).tolist()
results = collection.query(query_embeddings=question_emb, n_results=20)

context = " ".join(results["documents"][0])

client = MultiFMClient()

prompt = f"Given the following data, Please answer the question:  \n\n ##question \n {question}\n\n ##context \n {context}"

messages = [
    {
        "role": "system",
        "content": "You are a helpful agent, who answers with brevity. ",
    },
    {"role": "user", "content": prompt},
]


response = client.chat.completions.create(
    model="groq:llama3-70b-8192", messages=messages
)
print(response.choices[0].message.content)


response = client.chat.completions.create(
    model="anthropic:claude-3-opus-20240229", messages=messages
)
print(response.choices[0].message.content)


def chat(llm, messages):
    response = client.chat.completions.create(model=llm, messages=messages)
    return response.choices[0].message.content


llms = ["anthropic:claude-3-opus-20240229", "groq:llama3-70b-8192"]


# key word tagging
def keyword_tagging(doc, llms):
    prompt = f"Plese generate a list of 10-20 keywords seperated by commas from the following text:  \n\n ## Text \n {doc}\n\n "
    messages = [
        {
            "role": "system",
            "content": "You are a helpful agent, who generates a list of keywords seperated by commas and provides no other text. ",
        },
        {"role": "user", "content": prompt},
    ]
    ret = []
    for llm in llms:
        ret.append(chat(llm, messages))
    return ret


A, B = keyword_tagging(docs[3], llms)


def rewrite_as(doc, llms, style="cyberpunk author"):
    prompt = f"Please rewrite the following text as if it were written as a {style} author:  \n\n ## Text \n {doc}\n\n "
    messages = [
        {
            "role": "system",
            "content": "You are a helpful agent, who rewrites narative text wtih the same content and meaning but with a distinct voice and style. ",
        },
        {"role": "user", "content": prompt},
    ]
    ret = []
    for llm in llms:
        ret.append(chat(llm, messages))
    return ret


A, B = rewrite_as(docs[3], llms)
A[:200]
B[:200]
docs[3][:200]


def generic_compare(doc, llms, prompt="Translate the text to pirate"):
    prompt = f"{prompt}:  \n\n ## Text \n {doc}\n\n "
    messages = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": prompt},
    ]
    ret = []
    for llm in llms:
        ret.append(chat(llm, messages))
    return ret


results = {}
for doc in docs[:10]:
    results[doc] = generic_compare(doc, llms, "Write a 2 sentence summery")

best = []
for x in results.keys():
    print(f"Orignal text :  {x}\n\n")
    print(f"Option 1 text :  {results[x][0]}\n\n")
    print(f"Option 2  text :  {results[x][1]}\n\n")
    best.append(input("Which is best 1 or 2. 3 if indistinguisable:  "))
