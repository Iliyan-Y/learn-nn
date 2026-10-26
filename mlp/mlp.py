import torch
import matplotlib.pyplot as plt

from util import create_mapping


def run_mlp(words: list[str]):
    stoi, itos = create_mapping(words)
    print(itos)

    # MLP model 3 inputs -> hidden layer of xN neurons -> out of all words

    # build the dataset
    block_size = 3  # context length: how many characters do we take to predict the next one?

    # X = input to the n net
    # Y = labels for each example inside X
    X, Y = [], []
    for w in words:
        print(w)
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

    print("X", X)
    print("Y", Y)

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plt.plot(x, y, marker='o')
    plt.title("My First Terminal Plot")
    plt.show()
