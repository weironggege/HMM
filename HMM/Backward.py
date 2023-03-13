import numpy as np
from initt import init_param

def backward(A, B, O, PI, V):
    N, T = len(A), len(V)
    seen_idx = [O.index(idx) for idx in V]
    beta = np.zeros((T, N))
    for t in range(T-1, -1, -1):
        if t == T - 1:
            beta[t, :] = [1] * N
        else:
            for n in range(N):
                beta[t, n] = np.dot(A[n, :], beta[t+1,:] * B[:, seen_idx[t+1]])

    P = sum(beta[0, :] * B[:, seen_idx[0]] * PI)
    return beta, P

if __name__ == "__main__":
    _, O, PI, A, B, V = init_param()
    bet, prob = backward(A, B, O, PI, V)
    print(bet)
    print(prob)
