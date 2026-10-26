
import torch
import torch.nn.functional as F


def load_model(path="name_G_model.pth"):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    return checkpoint["W"], checkpoint["stoi"]


def score_name(name, W, stoi):
    name = name.lower().strip()

    if not name:
        raise ValueError("Name cannot be empty.")

    unknown_chars = set(name) - set(stoi.keys())
    unknown_chars.discard(".")

    if unknown_chars:
        raise ValueError(
            f"Unknown characters: {', '.join(sorted(unknown_chars))}"
        )

    # Include the start and end tokens.
    chars = ["."] + list(name) + ["."]

    log_probabilities = []

    with torch.no_grad():
        for current_char, next_char in zip(chars, chars[1:]):
            current_ix = stoi[current_char]
            next_ix = stoi[next_char]

            # Predict the next character.
            logits = W[current_ix]
            log_probs = F.log_softmax(logits, dim=0)

            # How likely was the actual next character?
            log_probabilities.append(log_probs[next_ix].item())

    # Average log probability per character transition.
    average_log_prob = sum(log_probabilities) / len(log_probabilities)

    # Geometric mean probability, expressed as a percentage.
    score = torch.tensor(average_log_prob).exp().item() * 100

    return score


if __name__ == "__main__":
    W, stoi = load_model()

    while True:
        name = input("Enter a name (or 'exit' to quit): ").strip()

        if name.lower() == "exit":
            break

        try:
            score = score_name(name, W, stoi)
            print(f"Model fit score for '{name}': {score:.2f}%\n")
        except ValueError as error:
            print(f"Error: {error}\n")
