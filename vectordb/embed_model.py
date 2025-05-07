import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("BM-K/KoSimCSE-roberta-multitask")
model = AutoModel.from_pretrained("BM-K/KoSimCSE-roberta-multitask").to(device)
model.eval()

class CustomEmbeddingFunction:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**batch, return_dict=False)
            embeddings = outputs[0][:, 0, :]  # CLS 토큰
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.__call__([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.__call__(texts)