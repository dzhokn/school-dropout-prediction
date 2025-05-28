import copy
import math
import pandas as pd             # Pandas is a library for data manipulation and analysis
import numpy as np              # Numpy is a library for numerical computing


def compute_y(x: np.ndarray, w: np.ndarray, b: float) -> float:
  """Calculates the predicted value of y for a given x (observation vector), w (weights), and b (bias)."""
  return b + w @ x

def compute_cost(X: pd.DataFrame, Y: np.ndarray, w: np.ndarray, b: float) -> float:
  """Calculates the error (cost) for a given X (set of vectors), Y (target values), w (weights), and b (bias).""" 
  m = X.shape[0]        # The number of examples (rows) in the training set
  cost = 0.
  X = X.values.tolist() # Conversion to lists is required to avoid issues with missing indices in the training set.
  Y = Y.tolist()        # Conversion to lists is required to avoid issues with missing indices in the training set.

  for i in range(m):
    cost += (compute_y(X[i], w, b) - Y[i]) ** 2
  return cost / m

def compute_gradient(X: pd.DataFrame, Y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]: 
    """
    Computes the gradient for linear regression with multiple features
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape               # (number of examples, number of features)
    X = X.values.tolist()       # Conversion to lists is required to avoid issues with missing indices in the training set.
    Y = Y.tolist()              # Conversion to lists is required to avoid issues with missing indices in the training set.
    dj_dw = np.zeros((n,))      # Now we return not a single weight, but a list of weights.
    dj_db = 0.

    for i in range(m):
        y_pred = compute_y(X[i], w, b)
        err = y_pred - Y[i]
        # Extra loop: we have `n` features (w1, w2, ... wn) - we assess how strongly each of them affects the error (cost).
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i][j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m   # We divide all the weight derivatives by `m` (the number of rows)
    dj_db = dj_db / m   # We divide the bias derivative by `m` (the number of rows)

    return dj_db, dj_dw

def gradient_descent(X: pd.DataFrame, y: np.ndarray, w_in: np.ndarray, b_in: np.ndarray, alpha: float, num_iters: int) -> tuple[np.ndarray, float, list]: 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    J_history = []            # An array to store cost J and w's at each iteration primarily for graphing later
    w = copy.deepcopy(w_in)   # Avoid modifying global `w_in within current function
    b = b_in                  # Avoid modifying global `b_in within current function

    for i in range(num_iters):
        # Calculate the gradient for these `w` and `b`
        dj_db, dj_dw = compute_gradient(X, y, w, b)

        # Update `w` and `b` based on the gradient (up or down) and the specified learning rate
        w = w - alpha * dj_dw   # vector operation (w and dj_dw are vectors with the same size)
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            J_history.append( compute_cost(X, y, w, b) )

        # Print the cost after every 10% of the iterations
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history # return final w,b and J history for graphing
