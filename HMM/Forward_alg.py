import numpy as np
from initt import init_param

def forward(A, B, O, PI, V):
    N, T = len(A), len(V)
    seen_idx = [O.index(idx) for idx in V]

    alpha = np.zeros((T, N))

    for t in range(T):
        if t == 0:
            alpha[t, :] = PI * B[:, seen_idx[t]]
        else:
            for n in range(N):
                alpha[t, n] = np.dot(alpha[t-1,:], A[:,n])
            alpha[t,:] *= B[:, seen_idx[t]]
    P = sum(alpha[T-1, :])
    return alpha, P

if __name__ == "__main__":
    _, O, PI, A, B, V = init_param()
    alh, prob = forward(A, B, O, PI, V)
    print(alh)
    print(prob)