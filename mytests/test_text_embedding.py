from litellm import embedding
import os
from openai import OpenAI


# client = OpenAI(
#     api_key=api_key,
#     base_url=base_url,
# )
# completion = client.embeddings.create(
#     model="text-embedding-v4",
#     input='衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买',
#     dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
#     encoding_format="float"
# )
# print(completion.model_dump_json())


# embedding_model = "openai/text-embedding-v4"
# api_key = os.getenv("DASHSCOPE_API_KEY")
# base_url = os.getenv("DASHSCOPE_API_BASE")

# response = embedding(
#     model=embedding_model,
#     input=["good morning from litellm","我爱你，中国"],
#     api_base=base_url,
#     api_key=api_key
# )
# print(f"Embedding:{response}")



embedding_model = "openai/bge-large-zh-v1.5"
api_key = os.getenv("GITEE_API_KEY")
base_url = os.getenv("GITEE_API_BASE")

response = embedding(
    model=embedding_model,
    input=["good morning from litellm","我爱你，中国"],
    api_base=base_url,
    api_key=api_key
)
print(f"Embedding:{response}")
