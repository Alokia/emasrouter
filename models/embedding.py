from sentence_transformers import SentenceTransformer
import torch


def get_sentence_embedding(sentence):
    model = SentenceTransformer('/home/alokia/Research/Model_CheckPoint/sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)


class SentenceEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('/home/alokia/Research/Model_CheckPoint/sentence-transformers/all-MiniLM-L6-v2')

    def forward(self, sentence):
        # [batch_size, 384]
        embeddings = self.model.encode(sentence)
        return torch.tensor(embeddings)
