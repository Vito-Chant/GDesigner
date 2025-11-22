from sentence_transformers import SentenceTransformer

# 将模型初始化移到全局作用域，只加载一次
_EMBEDDING_MODEL = None

def get_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        # 可以在这里指定 device，例如 device='cuda' 或 device='cpu'
        _EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _EMBEDDING_MODEL

def get_sentence_embedding(sentence):
    model = get_model()
    embeddings = model.encode(sentence)
    return embeddings