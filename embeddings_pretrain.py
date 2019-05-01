import torch
from torch import nn
import math
import numpy as np
import io
from dictionaries import IndexDictionary


def create_weight(index2word, embedding):
    emb_dim = 300
    words_found = 0
    wnf = []
    matrix_len = len(index2word.keys())
    weight_matrix= np.zeros((matrix_len, emb_dim))
    for k,v in index2word.items():
        if v in embedding:
            weight_matrix[k] = embedding[v]
            words_found += 1
        else:
            weight_matrix[k] = np.random.normal(size=(emb_dim, ))
            wnf.append(k)
    return weight_matrix, wnf, words_found

def load_embedding(fname, words_to_load=np.inf):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for index, line in enumerate(fin):
        if index > words_to_load:
            break
        else:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = [float(i) for i in tokens[1:]]
    return data




class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout_prob (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, num_embeddings, embedding_dim, dim, dropout_prob=0., padding_idx=0, max_len=5000, pretrain=True):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        
        if pretrain:
            """
            Load pretrained embedding 
            """
            fname_eng ='/scratch/wz1218/zh/fasttext300d.vec'
            embedding_mat = load_embedding(fname_eng)

            """
            Index2word
            """
            with open("/scratch/wz1218/Transformer-pytorch/data/example/processed/vocabulary-source.txt") as file:
                vocab_tokens = {}
                for line in file:
                    vocab_index, vocab_token, count = line.strip().split('\t')
                    vocab_index = int(vocab_index)
                    vocab_tokens[vocab_index] = vocab_token
            """
            create weight matrix
            """

            weight_matrix, _, _= create_weight(vocab_tokens, embedding_mat) 
            embed_mat = torch.from_numpy(weight_matrix).float()
            self.num_embeddings, self.embedding_dim = embed_mat.shape
            self.embedding = nn.Embedding.from_pretrained(embed_mat, freeze = True)
            print("pretrain embedding has been loaded")
        else:
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.weight = self.embedding.weight
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim

    def forward(self, x, step=None):
        x = self.embedding(x)
        x = x * math.sqrt(self.dim)
        if step is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, step]
        x = self.dropout(x)
        return x
