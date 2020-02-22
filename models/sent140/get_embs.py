import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('-f',
                    help='path to .txt file containing word embedding information;',
                    type=str,
                    default='glove.6B.50d.txt')
parser.add_argument('-o',
                    help='name of output file;',
                    type=str,
                    default='embs.json')

args = parser.parse_args()

dim = 300 if args.f == 'glove.6B.300d.txt' else 50
print('Starting')

# lines = []
with open(args.f, 'r') as inf:
    lines = inf.readlines()
lines = [l.split() for l in lines]
vocab = [l[0] for l in lines]
emb_floats = [[float(n) for n in l[1:]] for l in lines]
emb_floats.append([0.0 for _ in range(dim)])  # for unknown word
js = {'vocab': vocab, 'emba': emb_floats}
with open(args.o, 'w') as ouf:
    json.dump(js, ouf)
print('Done')
