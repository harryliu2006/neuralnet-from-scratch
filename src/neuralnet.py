
import numpy as np

def init_parameters(layer_sizes, seed=42):
    rng = np.random.default_rng(seed)
    L = len(layer_sizes) - 1
    W, B = {}, {}
    for l in range(1, L + 1):
        W[l] = rng.standard_normal((layer_sizes[l], layer_sizes[l - 1]))
        B[l] = rng.standard_normal((layer_sizes[l], 1))
    return W, B

def standardize_train_val(X_train, X_val):
    mu = X_train.mean(axis=1, keepdims=True)
    sigma = X_train.std(axis=1, keepdims=True) + 1e-8
    return (X_train - mu) / sigma, (X_val - mu) / sigma

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z): 
    return np.maximum(0,z)

def relu_derivative(z): 
    return (z>0).astype(float)

def feed_forward(X, W, B):
    A, Z = {0: X}, {}
    L = len(W)
    for l in range(1, L + 1):
        Z[l] = W[l] @ A[l - 1] + B[l]
        if l == L: 
            A[l] = sigmoid(Z[l])
        else: 
            A[l] = relu(Z[l])
    return A, Z

def bce_loss(Y_hat, Y, eps=1e-12):
    Y_hat = np.clip(Y_hat, eps, 1 - eps)
    m = Y.shape[1]
    losses = -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return float(np.sum(losses) / m)

def backprop(A, Z, Y, W):
    grads = {}
    L = len(W)
    m = Y.shape[1]

    #output layer
    dZ = (A[L] - Y) / m
    grads[f"dC_dW{L}"] = dZ @ A[L - 1].T
    grads[f"dC_db{L}"] = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W[L].T @ dZ

    #hidden layers
    for l in range(L - 1, 0, -1):
        dZ = dA_prev * relu_derivative(Z[l])
        grads[f"dC_dW{l}"] = dZ @ A[l - 1].T
        grads[f"dC_db{l}"] = np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA_prev = W[l].T @ dZ
    return grads

def update(W, B, gradients, lr=0.1):
    for l in range(1, len(W) + 1):
        W[l] -= lr * gradients[f"dC_dW{l}"]
        B[l] -= lr * gradients[f"dC_db{l}"]
    return W, B

def train_val_split(X, Y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    m = X.shape[1]
    idx = rng.permutation(m)
    val_sz = int(m * val_ratio)
    val_idx, train_idx = idx[:val_sz], idx[val_sz:]
    return X[:, train_idx], Y[:, train_idx], X[:, val_idx], Y[:, val_idx]

def predict(W, B, X, thr=0.5):
    A, _ = feed_forward(X, W, B)
    return (A[len(W)] >= thr).astype(int)

def accuracy(W, B, X, Y, thr=0.5):
    return float((predict(W, B, X, thr) == Y).mean())

def train_early_stop(layer_sizes, X_train, Y_train, X_val, Y_val,
                     epochs=10000, lr=0.1, patience=20, min_delta=1e-4, print_every=100):
    W, B = init_parameters(layer_sizes)
    best_W, best_B = None, None
    best_val_loss = np.inf
    bad_epochs = 0
    L = len(W)

    for epoch in range(epochs):
        A_train, Z_train = feed_forward(X_train, W, B)
        J_train = bce_loss(A_train[L], Y_train)
        grads = backprop(A_train, Z_train, Y_train, W)
        W, B = update(W, B, grads, lr)

        A_val, _ = feed_forward(X_val, W, B)
        J_val = bce_loss(A_val[L], Y_val)

        if epoch % print_every == 0:
            tr_acc = accuracy(W, B, X_train, Y_train)
            va_acc = accuracy(W, B, X_val, Y_val)
            print(f"epoch {epoch:4d} | train_loss={J_train:.4f} | val_loss={J_val:.4f} "
                  f"| train_acc={tr_acc:.2f} | val_acc={va_acc:.2f}")

        if best_val_loss - J_val > min_delta:
            best_val_loss = J_val
            best_W = {l: W[l].copy() for l in W}
            best_B = {l: B[l].copy() for l in B}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if best_W is not None:
                    for l in W: W[l] = best_W[l]
                    for l in B: B[l] = best_B[l]
                break
    return W, B
