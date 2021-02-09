import argparse as ap
import numpy as np
import sys
import random as rm

parser = ap.ArgumentParser()
parser.add_argument("-s", "--square", type=int, nargs = 1)
parser.add_argument("-rc", "--rows-columns", nargs = 2, type = int)

args = parser.parse_args(sys.argv[1:])

if args.square:
    width = args.square[0]
    a = np.random.uniform(low = -50.0, high = 75.0, size = (width, width))
    b = np.random.uniform(low = -75.0, high = 50.0, size = (width, width))
    np.savetxt('asq.txt', a, fmt = "%f")
    np.savetxt('bsq.txt', b, fmt = "%f")
    c = np.matmul(a, b)
    np.savetxt('csq.txt', c, fmt = "%f")

elif args.rows_columns:
    height = args.rows_columns[0]
    width = args.rows_columns[1]
    a = np.random.uniform(low = -50.0, high = 75.0, size = (height, width))
    b = np.random.uniform(low = -75.0, high = 50.0, size = (width, height))
    print(f"Size of A: {a.size}, Size of B: {b.size}")
    np.savetxt('a.txt', a, fmt = "%f")
    np.savetxt('b.txt', b, fmt = "%f")
    c = np.matmul(a, b)
    np.savetxt('c.txt', c, fmt = "%f")


