def create_mapping(words: list[str]):
    # build the vocabulary of characters and mappings to/from integers
    # Create alphabet object programmatically
    chars = sorted(list(set(''.join(words))))
    alphabetIndex = {s: i + 1 for i, s in enumerate(chars)}
    alphabetIndex['.'] = 0
    indexLetter = {i: s for s, i in alphabetIndex.items()}
    return alphabetIndex, indexLetter
