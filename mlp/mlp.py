import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

from util import create_mapping


def run_mlp(words: list[str]):
    g = torch.Generator().manual_seed(2147483647)
    stoi, itos = create_mapping(words)
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    total_number_of_possible_outputs = len(itos)  # 27
    # MLP model 3 inputs -> hidden layer of xN neurons -> out of all words
    block_size = 3  # context length: how many characters do we take to predict the next one?
    dimension = 10  # embeding the params
    h_neurons = 300
    # this can also be the value from emb.shape torch.Size([32, 3, 2])  so
    # we can do emb.shape[1] * emb.shape[2]
    h_dim = block_size * dimension

    # indexing all 27 characters simmilar to bigram F.once_hot encoding
    # creating the weights for the first lyaer of the nn input
    C = torch.randn(total_number_of_possible_outputs, dimension, generator=g, requires_grad=True)

    Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)
    Xte, Yte = build_dataset(words[n2:], block_size, stoi)

    emb = C[Xtr]
    # Layer 1
    W1 = torch.randn(h_dim, h_neurons, generator=g, requires_grad=True)
    B1 = torch.randn(h_neurons, generator=g, requires_grad=True)

    # Layer 2 input should be the out of our pev layer and out should be the number of possible outputs
    W2 = torch.randn(h_neurons, total_number_of_possible_outputs, generator=g, requires_grad=True)
    b2 = torch.randn(total_number_of_possible_outputs, generator=g, requires_grad=True)

    params = [C, W1, B1, W2, b2]
    print(sum(p.nelement() for p in params))

    learning_rate = torch.linspace(-3, 0, 1000)  # 0.001 to 1
    lrs = 10 ** learning_rate  ## but exp
    lri = []
    lossi = []
    stepi = []

    samples = 32 * 3
    # forward pass
    for i in range(200000):
        mini_batch = torch.randint(0, Xtr.shape[0], (samples,))
        emb = C[Xtr[mini_batch]]
        # -1 will infer whatever the value from emb.shape[0] is
        h = torch.tanh(emb.view(-1, h_dim) @ W1 + B1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[mini_batch])
        # print("loss ->", loss.item())

        # Backwards
        for p in params:
            p.grad = None
        loss.backward()
        # lr = lrs[i]
        lr = -1 / 10 if i < 100000 else -0.1 / 10
        for p in params:
            p.data += lr * p.grad

        # Track
        # lri.append(lr)
        # lri.append(learning_rate[i])
        # lossi.append(loss.item())

    # print("loss ->", loss.item())

    total_n_params = sum(p.nelement() for p in params)  # number of parameters in total
    print("params ->", total_n_params)
    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, h_dim) @ W1 + B1)
    logits = h @ W2 + b2
    total_loos = F.cross_entropy(logits, Ydev)
    print("loss totla ->", total_loos.item())

    # plt.plot(lri, lossi)
    # plt.show()

    def samle():
        # sample from the model
        g = torch.Generator().manual_seed(2147483647 + 10)

        for _ in range(20):
            out = []
            context = [0] * block_size  # initialize with all ...
            while True:
                emb = C[torch.tensor([context])]  # (1,block_size,d)
                h = torch.tanh(emb.view(1, -1) @ W1 + B1)
                logits = h @ W2 + b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break

            print(''.join(itos[i] for i in out))

    samle()


def build_dataset(words: list[str], block_size: int, stoi: dict):
    # X = input to the n net
    # Y = labels for each example inside
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            # Convert character to number
            ix = stoi[ch]
            # Store the previous 3(n) characters
            X.append(context)
            # Store the character to predict
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            # Slide the context forward
            context = context[1:] + [ix]  # crop and append

    # dataset
    Xtr = torch.tensor(X)
    Ytr = torch.tensor(Y)
    print(Xtr.shape, Ytr.shape)
    return Xtr, Ytr
