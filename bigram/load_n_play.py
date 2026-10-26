
import torch
import torch.nn.functional as F


def load_model(path="name_G_model.pth"):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    return (
        checkpoint["W"],
        checkpoint["stoi"],
        checkpoint["itos"],
    )


def generate_name(W, stoi, itos, temperature=1.0, start=""):
    # Start at the special beginning token unless
    # the user specifies a starting character.
    ix = stoi[start.lower()] if start else 0
    name = start.lower()

    with torch.no_grad():
        while True:
            # The current character becomes a one-hot vector.
            xenc = F.one_hot(
                torch.tensor([ix]),
                num_classes=W.shape[0],
            ).float()

            # Predict the next character.
            logits = xenc @ W

            # Temperature controls how random the output is.
            logits = logits / temperature
            probs = F.softmax(logits, dim=1)

            # Sample the next character.
            ix = torch.multinomial(probs, num_samples=1).item()

            # Index 0 means the name has ended.
            if ix == 0:
                break

            name += itos[ix]

    return name


if __name__ == "__main__":
    W, stoi, itos = load_model()

    for _ in range(8):
        print(generate_name(W, stoi, itos, temperature=1.1, start="I"))
