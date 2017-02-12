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

def output_gradient(y, t):
    """
    Compute the gradient of the MSE cost function with respect to the output y
    """
    return 2.0 * (y - t) / nb_of_samples

def backward_gradient(X, S, grad_out, wRec):
    """
    Backpropagate the gradient computed at the output (grad_out) through the network.
    Accumlate the parameter gradients for wX and wRec by each laher by addtion.
    Return the parameter gradients as a tuple, and the gradients at the toutput of eatch layer.
    """
    # Initialize the array that stores the gradient
    grad_over_time = np.zeros((X.shape[0], X.shape[1] + 1))
    grad_over_time[:,-1] = grad_out
    # Set the gradient accumulations to 0
    wx_grad = 0
    wRec_grad = 0
    for k in range(X.shape[1], 0, -1):
        # Computed the parameter gradients and accumulate the results.
        wx_grad += np.sum(grad_over_time[:,k] * X[:,k -1])
        wRec_grad += np.sum(grad_over_time[:,k] * S[:,k-1])
        # Compute the gradient at the otout of the previous layer
        grad_over_time[:,k-1] = grad_over_time[:,k] * wRec
    return (wx_grad, wRec_grad), grad_over_time

def update_rprop(X, t, W, W_prev_sign, W_delta, eta_p, eta_n):
    """
    Update Rprop values in one iteration.
    X: input data.
    t: targets.
    W: Current weight parameters.
    W_prev_sign: Previous sign of the W gradient.
    W_delta: Rprp update values (Delta).
    eta_p, eta_n: Rprp hyper parametrers.
    """
    # Perform forward and backward pass to get the gradients
    S = forward_states(X, W[0], W[1])
    grad_out = output_gradient(S[:,-1], t)
    W_grads, _ = backward_gradient(X, S, grad_out, W[1])
    W_sign = np.sign(W_grads) # Sign of new gradient
    # Update the Delta (update value) for ewach weight paramter seperately
    for i, _ in enumerate(W):
        if W_sign[1] == W_prev_sign[i]:
            W_delta[i] *= eta_p
        else:
            W_delta[i] *= eta_n
    return W_delta, W_sign

nb_of_samples = 20
sequence_len = 10

# Creat input stream randomly
X = np.zeros((nb_of_samples, sequence_len))
for row_idx in range(nb_of_samples):
    # around is round func
    X[row_idx, :] = np.around(np.random.rand(sequence_len)).astype(int)

# this is correcgt answer
t = np.sum(X, axis=1)

eta_p = 1.2
eta_n = 0.5

W = [-1.5, 2] # [wx, wRec]
W_delta = [0.001, 0.001]
W_sign = [0, 0]

ls_of_ws = [(W[0], W[1])]

for i in range(500):
    W_delta, W_sign = update_rprop(X, t, W, W_sign, W_delta, eta_p, eta_n)
    for i, _ in enumerate(W):
        W[i] -= W_sign[i] * W_delta[i]
    ls_of_ws.append((W[0], W[1]))

print('Final weights are', W)
