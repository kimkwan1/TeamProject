import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_spin_grid(fname):

    spins = []
    with open(fname, 'r') as fid:
        line = fid.readline()
        while line:
            tmp = np.array([int(i) for i in line.split(',')[:-1]], dtype=int)
            n = int(np.sqrt(tmp.shape[0]))
            spins.append(tmp.reshape(n, n))
            line = fid.readline()
    
    return spins


if __name__ == "__main__":
    
    T = [i*0.01 for i in range(501)]
    
    for t in tqdm(T):
        spins_all = []
        for n in range(5):
            fname = "spins_%.2f_%d.csv"%(t, n)
            spins = load_spin_grid(fname)
            spins_all.append(spins)
        spins_all = np.array(spins_all, dtype=int)

        with open("spins_%.2f.pkl"%(t), "wb") as fid:
            pkl.dump(spins_all, fid)

    print("Done")
        
            

    
    
    
