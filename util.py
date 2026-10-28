import torch


def create_mapping(words: list[str]):
    # build the vocabulary of characters and mappings to/from integers
    # Create alphabet object programmatically
    chars = sorted(list(set(''.join(words))))
    alphabetIndex = {s: i + 1 for i, s in enumerate(chars)}
    alphabetIndex['.'] = 0
    indexLetter = {i: s for s, i in alphabetIndex.items()}
    return alphabetIndex, indexLetter


def build_dataset(words: list[str], block_size, stoi):
    X, Y = [], []
    for w in words:

        # print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y
