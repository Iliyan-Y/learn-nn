import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from util import create_mapping, build_dataset


def mlp_lecture(words: list[str]):
    stoi, itos = create_mapping(words)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    total_number_of_possible_outputs = len(itos)  # 27
    # MLP model 3 inputs -> hidden layer of xN neurons -> out of all words
    block_size = 3  # context length: how many characters do we take to predict the next one?
    dimension = 2  # embeding the params
    h_neurons = 300
    # this can also be the value from emb.shape torch.Size([32, 3, 2])  so
    # we can do emb.shape[1] * emb.shape[2]
    h_dim = block_size * dimension

    Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)
    Xte, Yte = build_dataset(words[n2:], block_size, stoi)

    g = torch.Generator().manual_seed(2147483647)  # for reproducibility
    C = torch.randn((27, 10), generator=g)
    W1 = torch.randn((30, 200), generator=g)
    b1 = torch.randn(200, generator=g)
    W2 = torch.randn((200, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]
    n_params = sum(p.nelement() for p in parameters)

    for p in parameters:
        p.requires_grad = True

    lre = torch.linspace(-3, 0, 1000)
    lrs = 10 ** lre
    lri = []
    lossi = []
    stepi = []

    for i in range(1000):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (32,))

        # forward pass
        emb = C[Xtr[ix]]  # (32, 3, 10)
        h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 200)
        logits = h @ W2 + b2  # (32, 27)
        loss = F.cross_entropy(logits, Ytr[ix])
        # print(loss.item())

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = lrs[i]
        # lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # track stats
        # lri.append(lre[i])
        stepi.append(i)
        lossi.append(loss.log10().item())

    # print(loss.item())
    plt.plot(stepi, lossi)
    emb = C[Xtr]  # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    loss = F.cross_entropy(logits, Ytr)
    print(loss)
