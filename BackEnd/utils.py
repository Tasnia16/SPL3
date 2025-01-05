import torch as th

def _get_first_singular_vectors_power_method(X, Y, max_iter=500, tol=1e-06):
    """
    Estimates the first singular vectors of matrices X and Y using the power method.

    Args:
        X (torch.Tensor): The first matrix.
        Y (torch.Tensor): The second matrix.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]: First singular vectors of X and Y, and number of iterations.
    """
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    eps = th.finfo(th.float64).eps

    # Initialize y_score to None and keep track of found valid column
    y_score = None

    # Try to find a valid y_score
    for col in Y.T:
        if th.any(th.abs(col) > eps):
            y_score = col
            print('Initial y_score shape:', y_score.shape)
            break

    if y_score is None:
        print("No valid column found in Y, returning early.")
        return None, None, 0  # Return early if no valid y_score is found

    x_weights_old = th.ones(X.shape[1])  # Initialize with a tensor of ones
    for i in range(max_iter):
        x_weights = th.matmul(X.T, y_score) / th.matmul(y_score, y_score)
        x_weights /= th.sqrt(th.matmul(x_weights, x_weights))
        x_score = th.matmul(X, x_weights)
        y_weights = th.matmul(Y.T, x_score) / th.matmul(x_score.T, x_score)
        y_score = th.matmul(Y, y_weights) / th.sqrt(th.matmul(y_weights, y_weights))

        print("Current x_score shape:", x_score.shape)
        x_weights_diff = x_weights - x_weights_old
        if th.matmul(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        x_weights_old = x_weights

    n_iter = i + 1

    # Placeholder for model training or other operations
    # You can call your model training function here
    print("Training model with obtained singular vectors...")
    # train_model(x_weights, y_weights)  # Uncomment and implement your model training logic

    return x_weights, y_weights, n_iter

def _get_first_singular_vectors_svd(X, Y):
    """
    Computes the first singular vectors of matrices X and Y using SVD.

    Args:
        X (torch.Tensor): The first matrix.
        Y (torch.Tensor): The second matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: First left and right singular vectors.
    """
    print("X shape for SVD:", X.shape)
    C = th.matmul(X.T, Y)
    U, _, Vt = th.linalg.svd(C)
    return U[:, 0], Vt[0, :]

def _svd_flip_1d(u, v):
    """
    Ensures that singular vectors u and v have consistent signs.

    Args:
        u (torch.Tensor): First singular vector.
        v (torch.Tensor): Second singular vector.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Adjusted singular vectors.
    """
    biggest_abs_val_idx = th.argmax(th.abs(u))
    sign = th.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign
    return u, v
