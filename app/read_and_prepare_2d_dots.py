import torch
from sklearn.manifold import TSNE
from pathlib import Path
from collections import namedtuple
Embeddings = namedtuple("Embeddings",["short", "long"])
def read_embeddings(path:Path,
                    long_filename="long.pt",
                    short_filename="short.pt"
                    )->namedtuple:
    short_embs = torch.load(path/short_filename)
    long_embs = torch.load(path/long_filename)


    return Embeddings(short_embs, long_embs)

def get_dots(embs, perplexity=5):
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(embs)
    x = X_tsne[:,1]
    y = X_tsne[:,0]
    return x, y

def main():
    path = Path("./data")
    embeddings = read_embeddings(path)
    short_dots = get_dots(embeddings.short)
    long_dots = get_dots(embeddings.long)
    torch.save(torch.Tensor(short_dots), path/"short_dots.pt")
    torch.save(torch.Tensor(long_dots), path/"long_dots.pt")

def get_all_dots(path:Path,
                long_filename="long.pt",
                short_filename="short.pt",
                perplexity=5):
    embeddings = read_embeddings(path,long_filename=long_filename,short_filename=short_filename)
    long_length = len(embeddings.long)
    embeddings = torch.cat((torch.Tensor(embeddings.long),torch.Tensor(embeddings.short)))
    dots = get_dots(embeddings,perplexity=perplexity)
    dots = torch.Tensor(dots)
    long_dots = dots[:,:long_length]
    short_dots = dots[:,long_length:]
    return Embeddings(short_dots,long_dots)


if __name__ == '__main__':
    main()
