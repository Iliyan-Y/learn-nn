from mlp.mlp import run_mlp

if __name__ == '__main__':
    words = open('names.txt', 'r').read().splitlines()
    # print(len(words))
    run_mlp(words)
