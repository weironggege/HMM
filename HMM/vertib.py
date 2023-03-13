import numpy as np
from initt import init_param



def viterbi(A, B, O, PI, V):
    N, T = len(A), len(V)
    seen_idx = [O.index(idx) for idx in V]
    theat, fi = np.zeros((T, N)), np.zeros((T, N))
    best_i = [0] * T
    for t in range(T):
        if t == 0:
            theat[t,:] = PI * B[:, seen_idx[t]]
        else:
            for n in range(N):
                theat[t, n] = np.max(theat[t-1,:] * A[:,n]) * B[n, seen_idx[t]]
                fi[t, n] = np.argmax(theat[t-1,:] * A[:,n])
    fi = fi.astype(int)
    best_i[T - 1] = np.argmax(theat[T-1, :])
    for t in range(T-2, -1, -1):
        best_i[t] = fi[t+1, best_i[t+1]]
    return fi, best_i

if __name__ == "__main__":
    _, O, PI, A, B, V = init_param()
    fi, best_idx = viterbi(A,B,O, PI, V)
    print(fi)
    print(best_idx)