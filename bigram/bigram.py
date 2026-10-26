import torch
import torch.nn.functional as F

from util import create_mapping


def run_bigram(words: list[str]):
    alphabetIndex, indexLetter = create_mapping(words)

    # create the training set of bigrams (x,y)
    xs, ys = [], []

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = alphabetIndex[ch1]
            ix2 = alphabetIndex[ch2]
            # print(ch1, ch2)
            xs.append(ix1)
            ys.append(ix2)

    # print(xs, ys)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    number_of_examples = xs.nelement()
    # print('number of examples: ', number_of_examples)

    # randomly initialize 27 neurons' weights. each neuron receives 27 inputs
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    iter = 120
    # gradient descent
    for k in range(iter):
        # ------------
        # Forward pass
        # ------------
        # One -hot encoding https://www.youtube.com/watch?v=G2iVj7WKDFk
        # convert the letter numerical index like emma [0, 5, 13, 13, 1] into numerical values that can be assigned to the nn to understand
        xenc = F.one_hot(xs, num_classes=27).float()  # input to the network: one-hot encoding
        # we are adding random weight to the data set to fulfil activation x*w
        logits = xenc @ W  # predict log-counts

        # Softmax turns a vector of raw numbers (called logits) into a probability distribution that adds up to 1 (or 100%).
        counts = logits.exp()  # counts, equivalent to N
        probs = counts / counts.sum(1, keepdim=True)  # probabilities for next character
        # print(probs.shape)

        # L2 Regularisation (often implemented as Weight Decay).
        # to stop neural network from over-memorising the training data (overfitting) and to keep the weights from exploding.
        lossSmoother = 0.01 * (W ** 2).mean()
        # Because Gradient Descent's only goal is to make the Total Loss as close to 0 as possible,
        # it forces a tug-of-war: it wants to predict accurately, but it also wants to keep the weights (\(W\)) as small and close to 0 as possible.
        loss = -probs[torch.arange(number_of_examples), ys].log().mean() + lossSmoother

        if k % 20 == 0:
            print(f"Epoch {k:4d} | Loss: {loss.item():.6f}")

        # ------------
        # Backward pass
        # ------------
        W.grad = None  # set to zero the gradient
        loss.backward()

        # update
        W.data += -50 * W.grad

    checkpoint = {
        "W": W.detach().clone(),
        "alphabet": chars,
        "stoi": alphabetIndex,
        "itos": indexLetter,
    }

    torch.save(checkpoint, "bigram/name_G_model.pth")
    print("Model saved!")
    # g = torch.Generator().manual_seed(2147483647)
    # for i in range(5):
    #     out = []
    #     ix = 0
    #     while True:
    #
    #         # ----------
    #         xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    #         logits = xenc @ W  # predict log-counts
    #         counts = logits.exp()  # counts, equivalent to N
    #         p = counts / counts.sum(1, keepdim=True)  # probabilities for next character
    #         # ----------
    #
    #         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    #         out.append(indexLetter[ix])
    #         if ix == 0:
    #             break
    #     print(''.join(out))
