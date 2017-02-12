import numpy as np

# http://peterroelants.github.io/posts/rnn_implementation_part01/
#
# Traing RNN to count 1 in stream

# Forward step functions
def update_state(xk, sk, wx, wRec):
    """
    Compute state k from previous sate sk and current input xk,
    by use of the input weights (wx) and recursive weights (wRec).
    """
    return xk * wx + sk * wRec

def forward_states(X, wx, wRec):
    """
    Unfold the network and compute all state activations given the input X,
    and input weiths (wx) and recursive weights (wRec).
    Return the state activations in a matrix, the last column S[:, -1] contains final activations.
    """
    # initialize the matrix that holds all state for all input sequences.
    # The initial state s0 is set to 0
    S = np.zeros((X.shape[0], X.shape[1] + 1))

    # Use the reccurrence relation defined by update_state to update the
    # states through time
    for k in range(0, X.shape[1]):
        # S[k] = S[k - 1] * wRec + X[k] * wx
        S[:, k + 1] = update_state(X[:,k], S[:,k], wx, wRec)
    return S

def cost(y, t):
    """
    Return the MSE between the targets tan the outputs y.
    """
    return ((t - y) ** 2).sum() / nb_of_samples

nb_of_samples = 20
sequence_len = 10

# Creat input stream randomly
X = np.zeros((nb_of_samples, sequence_len))
for row_idx in range(nb_of_samples):
    # around is round func
    X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)

# this is correcgt answer
t = np.sum(X, axis=1)


