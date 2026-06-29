from chromadb.api.collection_configuration import collection_configuration_to_json

md5_path = "./md5.text"


#Chroma
collection_name = "rag"
persist_directory = "./chroma__db"


#spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n",  ".", "!", "?", " ", "  "]
max_split_char_number = 1000

#
similarity_threshold =  2    #检索返回的文档数

embedding_model_name = "text-embedding-v3"
chat_model_name = "qwen3-max"


session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }