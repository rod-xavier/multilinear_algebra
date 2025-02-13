import numpy as np
from typing import Union
from scipy.linalg import svd
from scipy.linalg import khatri_rao
from scipy.optimize import least_squares
from tensorly.tenalg import multi_mode_dot
from scipy.linalg import khatri_rao as kr

# From Homework 1
def hadamard_product(matrix_a, matrix_b):
    """
    Computes the Hadamard product (element-wise multiplication) of two matrices, matrix_a and matrix_b.

    Parameters:
    matrix_a (list of list of int/float): A 2D list (matrix) with dimensions (m x n).
    matrix_b (list of list of int/float): A 2D list (matrix) with dimensions (m x n).

    Returns:
    list of list of int/float: The resulting matrix from the Hadamard product of matrix_a and matrix_b with dimensions (m x n).

    Raises:
    ValueError: If the matrices have different dimensions.

    Example:
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    result = hadamard_product(matrix_a, matrix_b)
    print(result)
    """
    
    # Check if matrices have the same dimensions
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError("Matrices must have the same dimensions for Hadamard product.")
    
    # Get the dimensions of the matrices (m x n)
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])

    # Initialize the result matrix with the same dimensions as input matrices
    result_matrix = [[None for _ in range(cols_a)] for _ in range(rows_a)]

    # Perform the element-wise multiplication (Hadamard product)
    for i in range(rows_a):
        for j in range(cols_a):
            result_matrix[i][j] = matrix_a[i][j] * matrix_b[i][j]
    
    return result_matrix
    
def kron_product(matrix_a, matrix_b):
    """
    Computes the Kronecker product of two matrices, matrix_a and matrix_b.

    Parameters:
    matrix_a (list of list of int/float): A 2D list (matrix) with dimensions (m x n).
    matrix_b (list of list of int/float): A 2D list (matrix) with dimensions (p x q).

    Returns:
    list of list of int/float: The resulting matrix from the Kronecker product of matrix_a and matrix_b with dimensions (mp x nq).

    Raises:
    ValueError: If any matrix has rows of inconsistent lengths.

    Example:
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    result = kronecker_product(matrix_a, matrix_b)
    print(result)
    """
    
    # Validate matrix_a
    length_a = len(matrix_a[0])
    for row in matrix_a:
        if len(row) != length_a:
            raise ValueError("Matrix A has inconsistent row lengths.")
    
    # Validate matrix_b
    length_b = len(matrix_b[0])
    for row in matrix_b:
        if len(row) != length_b:
            raise ValueError("Matrix B has inconsistent row lengths.")
    
    # Get shape of matrix A (m x n) and matrix B (p x q)
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    # Resulting matrix dimensions (mp x nq)
    results_matrix = [[None for _ in range(cols_a * cols_b)] for _ in range(rows_a * rows_b)]

    # Perform the Kronecker product
    for i in range(rows_a):
        for j in range(cols_a):
            # Multiply current element of matrix_a with all elements of matrix_b
            submatrix = [[element * matrix_a[i][j] for element in row] for row in matrix_b]

            # Calculate placement offsets
            row_offset = i * rows_b
            col_offset = j * cols_b

            # Place the submatrix in the result matrix
            for m in range(rows_b):
                for n in range(cols_b):
                    results_matrix[row_offset + m][col_offset + n] = submatrix[m][n]
    
    return results_matrix

def kr_product(matrix_a, matrix_b):
    """
    Computes the Khatri-Rao product (column-wise Kronecker product) of two matrices, matrix_a and matrix_b.

    Parameters:
    matrix_a (list of list of int/float): A 2D list (matrix) with dimensions (m x n).
    matrix_b (list of list of int/float): A 2D list (matrix) with dimensions (p x n).

    Returns:
    list of list of int/float: The resulting matrix from the Khatri-Rao product of matrix_a and matrix_b with dimensions (mp x n).

    Raises:
    ValueError: If the number of columns in matrix_a and matrix_b do not match.

    Example:
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]
    result = khatri_rao_product(matrix_a, matrix_b)
    print(result)
    """
    
    # Validate matrix_a and matrix_b
    cols_a = len(matrix_a[0])
    cols_b = len(matrix_b[0])
    
    if cols_a != cols_b:
        raise ValueError("Matrix A and Matrix B must have the same number of columns.")
    
    # Get the number of rows for both matrices
    rows_a = len(matrix_a)
    rows_b = len(matrix_b)

    # Resulting matrix dimensions (mp x n)
    results_matrix = []
    
    # Perform the Khatri-Rao product column by column
    for j in range(cols_a):
        # Extract the j-th column of A and B
        col_a = [matrix_a[i][j] for i in range(rows_a)]
        col_b = [matrix_b[i][j] for i in range(rows_b)]
        
        # Perform the Kronecker product of the two columns
        column_result = [a * b for a in col_a for b in col_b]
        
        # Append the result to the result matrix
        results_matrix.append(column_result)
    
    # Transpose the results matrix to match the final shape (mp x n)
    final_result = [list(col) for col in zip(*results_matrix)]
    
    return final_result

# From Homework 3
def dB2linear(x: Union[int, np.ndarray]) -> any:
    """ Function to convert logarithmic scale to linear.
    Parameters:
    ---
    x : [scalar or 2-D array]
        Input data [dB].

    Returns:
    ---
    out: [scalar or 2-D array]
        Output data [linear (e.g.: watts)].
    """
    return 10**(x/10)

def alphaV(X: np.ndarray, snr_db: Union[int, float]):
    """Function to compute 'a' to control the noise of a signal."""
    V = np.random.normal(0, 1, X.shape) + 1j*np.random.normal(0, 1, X.shape)
    snr_linear = dB2linear(snr_db)
    alphaV_ = (np.linalg.norm(X, 'fro')**2)/snr_linear
    alpha = np.sqrt(alphaV_/(np.linalg.norm(X, 'fro')**2))
    return alpha*V
    
def LSKRF(X, A_shape, B_shape, SNR_dB):
    """
    Implements the LSKRF algorithm to estimate A and B.

    Args:
        X: Data matrix.
        A_shape: Shape of A.
        B_shape: Shape of B.
        SNR_dB: Signal-noise ratio for 
        additive noise term.

    Returns:
        A_hat, B_hat: Estimated matrices.
    """
    m, n = A_shape
    p, q = B_shape
    
    # Adding AWGN noise:
    if SNR_dB != None:
        X = X + alphaV(X, SNR_dB)
    else:
        pass

    # Inittializing empty matrices A_hat, B_hat and X_hat
    A_hat = A_hat = np.zeros(A_shape) + 1j*np.zeros(A_shape)
    B_hat = np.zeros(B_shape) + 1j*np.zeros(B_shape)
    X_hat = np.zeros(X.shape) + 1j*np.zeros(X.shape)

    # # Running the SVD algorithm at each column:
    for i in range(0, q): 
        # Reshaping the i-th column: 
        x = X[:, i].reshape(-1, 1, order='F')
        
        # Creating an unitary matrix from the i-th column:
        ## Running the unvec:
        ### NOTE: In order to perform the correct, 
        ### you must pass the " order='F' " on reshape from numpy.
        ### This method stacks the columns into rows correctly.
        Xc = x.reshape((p, m), order='F')
        
        # Running the SVD algorithm:
        U, Sigma, V = np.linalg.svd(Xc)
        
        # Taking the first eigenvalue (the strongest) from the sigma matrix
        # to obtain a the vectors a and b.
        ## The columns of U is equivalent to vector 'b'
        b = np.sqrt(Sigma[0])*U[:, 0]
        ## The rows of V is equivalent to vector 'a'
        a = np.sqrt(Sigma[0])*V[0, :]
        # Computing A and B hat:
        B_hat[:, i] = b
        A_hat[:, i] = a
        # Kronecker product of a and b to get x_hat
        ## The khatri-rao product is the columnwise kronecker...
        ## Let's stack into the X_hat columns
        X_hat[:, i] = np.kron(a, b) # Equivalent to "vec(outer(b, a))".
    return A_hat, B_hat

def NMSE(X0, X_hat):
    """
    Calculates the normalized mean square error (NMSE).

    Args:
        X0: Original data matrix.
        X_hat: Reconstructed data matrix.

    Returns:
        NMSE value.
    """
    return np.linalg.norm(X0 - X_hat)**2 / np.linalg.norm(X0)**2

# From Homework 4
def LSKronF(X, M, N, P, Q, SNR_dB):
    """
    Perform Least-Squares Kronecker Factorization.

    Parameters:
    - X: np.ndarray, the input matrix of shape (M*P, N*Q)
    - M, N: int, dimensions of A
    - P, Q: int, dimensions of B

    Returns:
    - A_hat: np.ndarray, estimated matrix A of shape (M, N)
    - B_hat: np.ndarray, estimated matrix B of shape (P, Q)
    """
    
    # Step 1: Rearrange X into a reshaped matrix X_tilde of shape (P*Q, M*N)
    X_tilde = np.zeros((P * Q, M * N), dtype=np.complex128)
    
    # Adding AWGN noise:
    if SNR_dB is not None:
        X = X + alphaV(X, SNR_dB)

    for m in range(M):
        for n in range(N):
            # Debug: Check block extraction
            block = X[m * P:(m + 1) * P, n * Q:(n + 1) * Q]

            if block.shape != (P, Q):
                raise ValueError(f"Block shape mismatch: {block.shape}, Expected: ({P}, {Q})")

            X_tilde[:, m * N + n] = block.flatten()

    # Step 2: Solve the least-squares problem using SVD
    U, S, Vt = svd(X_tilde, full_matrices=False)

    # Rank-1 approximation
    u1 = U[:, 0]  # First left singular vector
    v1 = Vt[0, :]  # First right singular vector
    sigma1 = S[0]  # Largest singular value

    # Step 3: Reconstruct A_hat and B_hat
    B_hat = np.sqrt(sigma1) * u1.reshape(P, Q)
    A_hat = np.sqrt(sigma1) * v1.reshape(M, N)

    return A_hat, B_hat

# From Homework 5
def kpsvd(X, M, N, P, Q, r=None):
    """
    Perform Kronecker Product Singular Value Decomposition (KPSVD).

    Parameters:
    - X: np.ndarray, the input matrix of shape (M*P, N*Q)
    - M, N: int, dimensions of the first factor
    - P, Q: int, dimensions of the second factor
    - r: int, optional, rank for nearest approximation (default: full rank)

    Returns:
    - U: list of np.ndarray, left singular matrices for each component
    - V: list of np.ndarray, right singular matrices for each component
    - Sigma: list of float, singular values
    """
    # Step 1: Rearrange X into a reshaped matrix X_tilde of shape (P*Q, M*N)
    X_tilde = np.zeros((P * Q, M * N), dtype=X.dtype)
    
    for m in range(M):
        for n in range(N):
            block = X[m * P:(m + 1) * P, n * Q:(n + 1) * Q]
            X_tilde[:, m * N + n] = block.flatten()

    # Step 2: Compute the SVD of X_tilde
    U_tilde, S, V_tilde_conj = svd(X_tilde, full_matrices=False)

    # Step 3: Define the rank for truncation (if r is None, use full rank)
    r = r if r is not None else min(P * Q, M * N)

    # Step 4: Extract the components
    U = []
    V = []
    Sigma = S[:r]
    
    for i in range(r):
        u_k = U_tilde[:, i]  # Left singular vector
        v_k = V_tilde_conj[i, :]  # Right singular vector (conjugate transpose row)
        
        # Reshape into desired dimensions
        U_k = u_k.reshape(P, Q)
        V_k = v_k.reshape(M, N)
        
        U.append(U_k)
        V.append(V_k)

    return U, V, Sigma

# From Homework 6
def unfold(X, mode):
    
    """
    Unfold a third-order tensor X into a matrix along the specified mode.

    Parameters:
    - X: numpy.ndarray
        Input tensor of shape (I, J, K).
    - mode: int
        The mode to unfold along (1, 2, or 3).

    Returns:
    - numpy.ndarray
        The unfolded matrix.
    """
    if mode == 1:
        return X.reshape(X.shape[0], -1, order='F')
    elif mode == 2:
        return X.transpose(1, 0, 2).reshape(X.shape[1], -1, order='F')
    elif mode == 3:
        return X.transpose(2, 0, 1).reshape(X.shape[2], -1, order='F')
    else:
        raise ValueError("Mode must be 1, 2, or 3.")
        
def fold(unfolded_matrix, original_shape, n_mode):
    """
    Reconstructs a 3D tensor from its n-mode unfolded matrix.

    Parameters:
        unfolded_matrix (numpy.ndarray): The unfolded matrix.
        original_shape (tuple): The shape of the original tensor.
        n_mode (int): The mode (1, 2, or 3) that was used for unfolding.

    Returns:
        numpy.ndarray: The reconstructed tensor.
    """
    if n_mode == 1:
        return unfolded_matrix.reshape(original_shape, order='F')  # 'F' for fu**ing Fortran order ¬¬"

    elif n_mode == 2:
    
        # 1. Reshape to (original_shape[1], -1, original_shape[2])
        temp = unfolded_matrix.reshape((original_shape[1], -1, original_shape[2]), order='F')
        
        # 2. Transpose to (original_shape[1], original_shape[0], original_shape[2])
        return temp.transpose(1, 0, 2)

    elif n_mode == 3:
        
        # 1. Reshape to (original_shape[2], original_shape[0], -1)
        temp = unfolded_matrix.reshape((original_shape[2], original_shape[0], -1), order='F')
        
        # 2. Transpose to (original_shape[0], original_shape[1], original_shape[2])
        return temp.transpose(1, 2, 0)

    else:
        raise ValueError("Mode must be 1, 2, or 3.")

# From Homework 7
def hosvd(X):
    """
    Performs Higher-Order Singular Value Decomposition (HOSVD) on a 
    third-order tensor.

    Args:
        X: The input tensor (3D numpy array).

    Returns:
        A tuple containing:
            S: The core tensor.
            U1: The matrix U^(1).
            U2: The matrix U^(2).
            U3: The matrix U^(3).
    """
    I1, I2, I3 = X.shape  # Get dimensions of input tensor X

    A1 = unfold(X, 1)  # Unfold X using first mode
    A2 = unfold(X, 2)  # Unfold X using second mode
    A3 = unfold(X, 3)  # Unfold X using third mode

    U1, _, _ = np.linalg.svd(A1)  # Compute the SVD of A1 and get U1
    U2, _, _ = np.linalg.svd(A2)  # Compute the SVD of A2 and get U2
    U3, _, _ = np.linalg.svd(A3)  # Compute the SVD of A3 and get u3

    S1 = np.dot(U1.T, np.dot(A1, np.kron(U3, U2)))  # Compute core tensor unfolded in the first mode
    S = fold(S1, (I1, I2, I3), 1)  # Fold the unfolded core tensor back to its original shape

    return S, U1, U2, U3  # Return core tensor S and factor matrices U1, U2, U3

def vec(matrix):
    return matrix.flatten(order='F').reshape(-1, 1)

def unvec(vector, nrow, ncol):
    return vector.reshape(nrow, ncol, order='F')

# From Homework 8
def hooi(X, ranks, n_iter=100000, tol=1e-6):
    """   
    Parameters:
    - X: The input tensor (numpy array).
    - ranks: Tuple specifying the desired multilinear rank.
    - n_iter: Number of iterations for optimization.
    - tol: Convergence tolerance.

    Returns:
    - core: The core tensor.
    - factors: List of factor matrices for each mode.
    """
    shape = X.shape
    N = len(shape)

    # Step 1: Initialize factor matrices using HOSVD
    factors = [np.linalg.svd(unfold(X, mode+1))[0][:, :ranks[mode]] for mode in range(N)]

    prev_core = None

    for _ in range(n_iter):
        for mode in range(N):
            # Compute mode-n product while leaving out mode-n matrix
            temp_tensor = multi_mode_dot(X, factors, transpose=True, skip=mode)
            
            # Compute SVD of mode-n unfolding
            U, _, _ = np.linalg.svd(unfold(temp_tensor, mode+1))
            factors[mode] = U[:, :ranks[mode]]  # Keep top singular vectors
        
        # Compute new core tensor
        core = multi_mode_dot(X, factors, transpose=True)

        # Check convergence
        if prev_core is not None and np.linalg.norm(core - prev_core) / np.linalg.norm(core) < tol:
            break
        prev_core = core

    X_hat = multi_mode_dot(core, factors)
    return X_hat

# From Homework 9
def mlskrf(A1, A2, A3, SNR_dB):
    """
    Performs Multidimensional LS-KRF algorithm.

    Args:
        - A1, A2, A3: Input tensors to generate X (NumPy array)
        - Signal to Noise Ratio value

    Returns:
        X: Tensor of dimensions (I1 * I2 * I3), R constructed from A1, A2, A3, represents original signal
        X_hat: Reconstructed tensor X_hat (NumPy array).
    """
    # Define dimensions (original dimensions)
    I1, I2, I3 , R = A1.shape[0], A2.shape[0], A3.shape[0], A1.shape[1]
    
    # Initialize X
    X = np.zeros((I1 * I2 * I3, R))
    
    # Initialize X_hat with correct dimensions
    X_hat = np.zeros((I1 * I2 * I3, R)) 

    # Estimating X from A1 A2 A3
    for r in range(R):
        kr_prod = np.kron(A1[:, r], np.kron(A2[:, r], A3[:, r]))
        X[:, r] = kr_prod.reshape(-1)  # Reshape to a column vector
    
    # Adding AWGN noise:
    if SNR_dB is not None:
        X = X + alphaV(X, SNR_dB)
    
    X = X.real

    # Initialize lists to store the hatimated factor matrices (original dimensions)
    A1_hat = np.zeros((I1, R))
    A2_hat = np.zeros((I2, R))
    A3_hat = np.zeros((I3, R))

    # Step 2: For columns r in column range of colums in X
    for r in range(R):
        # Step 2.1: Extract the r-th column of X
        x_r = X[:, r]

        # Step 2.2: Tensorize x_r to get X_r 
        X_r = x_r.reshape((I3, I2, I1), order='F')  # Reshape to original tensor form

        # Step 2.3: Compute the HOSVD of X_r
        S, U1, U2, U3 = hosvd(X_r)

        # Step 2.4: Extract and scale the first singular vectors
        y = np.sign(S[0, 0, 0])*(np.abs(S[0, 0, 0]) ** (1/3))
        a1_r = y * U3[:, 0]  # Size I1 (5)
        a2_r = y * U2[:, 0]  # Size I2 (4)
        a3_r = y * U1[:, 0]  # Size I3 (8)

        # Store the results in the hatimated factor matrices
        A1_hat[:, r] = a1_r  
        A2_hat[:, r] = a2_r  
        A3_hat[:, r] = a3_r

    # Estimating X_hat from A1_hat A2_hat A3_hat
    for r in range(R):
        kr_prod = np.kron(A1_hat[:, r], np.kron(A2_hat[:, r], A3_hat[:, r]))
        X_hat[:, r] = kr_prod.reshape(-1)  # Reshape to a column vector

    return X, X_hat

# From Homework 10
def mlskronf(matrix, shapes, known_values, SNR_dB):
    """
    Performs Multidimensional Least-Squares Kronecker Factorization on a 
    third-order tensor.

    Args:
        X: The input tensor (3D numpy array).
        shapes: The known shapes of A1 A2 A3 (list of tuples)
        known_values: A list of at least 1 value in A1 A2 A3 (python list)
        SNR_DB: noise to be added to X in dB (int)

    Returns:
        A1_hat
        A2_hat
        A3_hat
        X_hat
    """
    A1_row, A1_col = shapes[0]
    A2_row, A2_col = shapes[1]
    A3_row, A3_col = shapes[2]
    
    # Extracting Xn blocks
    nrow_a, ncol_a = shapes[0]
    nrow_b, ncol_b = [int(x / y) for x, y in zip(matrix.shape, shapes[0])]
    
    split = np.array(np.hsplit(matrix, ncol_a))

    aux = split.reshape(nrow_a * ncol_a, nrow_b, ncol_b)

    nrow_a, ncol_a = shapes[1]
    nrow_b, ncol_b = [int(x / y) for x, y in zip((nrow_b, ncol_b), shapes[1])]

    if np.iscomplexobj(matrix):
        out_x = np.zeros((nrow_b * ncol_b * nrow_a * ncol_a, np.prod(shapes[0])),
                         dtype=np.complex_)
    else:
        out_x = np.zeros((nrow_b * ncol_b * nrow_a * ncol_a, np.prod(shapes[0])))

    for s in range(aux.shape[0]):
        split_aux = np.array(np.hsplit(aux[s], ncol_a))
        split_aux = split_aux.reshape(nrow_a * ncol_a, nrow_b, ncol_b)
        out_x[:, [s]] = vec(split_aux.T.reshape(nrow_b * ncol_b, nrow_a * ncol_a))

    Xn = []
    for i in range(A1_row*A1_col):
        xi = out_x[:,i].reshape(A2_row*A3_row,A2_col*A3_col)
        Xn.append(xi)
        
    X_reshaped = np.concatenate([xn.reshape(-1) for xn in Xn]) # Vectorize and transpose each Xn
    X_reshaped = X_reshaped.reshape(A1_row*A1_col, A2_row*A2_col*A3_row*A3_col)
    
    # Creating tensor
    X_tensor = fold(X_reshaped.T, (A3_row*A3_col,A2_row*A2_col,A1_row*A1_col), 1)
    
    # Applying HOSVD
    S, U1, U2, U3 = hosvd(X_tensor)
    
    # Estimating A1 A2 A3
    A3_hat = U1[:,0].reshape(A3.shape[0],A3.shape[1], order='F')
    A2_hat = U2[:,0].reshape(A2.shape[0],A2.shape[1], order='F')
    A1_hat = U3[:,0].reshape(A1.shape[0],A1.shape[1], order='F')
    
    # Estimating scale factor based on known values of A1 A2 A3
    scale_facror_A1 = known_values[0] / A1_hat[0][0]
    scale_facror_A2 = known_values[1] / A2_hat[0][0]
    scale_facror_A3 = known_values[2] / A3_hat[0][0]
    
    # Approximating A1_hat A2_hat A3_hat
    A1_hat = A1_hat*scale_facror_A1
    A2_hat = A2_hat*scale_facror_A2
    A3_hat = A3_hat*scale_facror_A3
    
    X_hat = np.kron(A1_hat, np.kron(A2_hat, A3_hat))
    
    # Adding AWGN noise:
    if SNR_dB != None:
        X_hat = X_hat + alphaV(X_hat, SNR_dB)
    else:
        pass
    
    return A1_hat, A2_hat, A3_hat, X_hat

# From Homework 11
def als(X, rank, known_values, SNR_dB, n_runs=10000, delta=1e-6):
    """
    Performs Alternating Least Squares (ALS) decomposition on a 3-way tensor.

    This function decomposes a 3-way tensor `X` into three factor matrices 
    (A_hat, B_hat, C_hat) using the Alternating Least Squares (ALS) algorithm.
    It incorporates optional noise addition and scaling based on known values.

    Args:
        X (numpy.ndarray): The 3-way tensor to decompose.
        rank (int): The desired rank of the decomposition (number of components).
        known_values (list or tuple, optional): A tuple or list of three values. 
            These are used to scale the resulting factor matrices. 
        SNR_dB (float, optional): Signal-to-noise ratio in decibels. If provided, 
            additive white Gaussian noise (AWGN) is added to the tensor `X` before 
            decomposition. If None, no noise is added. Defaults to None.
        n_runs (int, optional): The maximum number of iterations for the ALS algorithm. 
            Defaults to 10000.
        delta (float, optional): The convergence tolerance. The algorithm stops when the 
            relative change in the reconstruction error between iterations is less than `delta`.
            Defaults to 1e-6.

    Returns:
        tuple: A tuple containing the scaled factor matrices (A_hat_scaled, B_hat_scaled, C_hat_scaled).
    """
    _, B_row, C_row = X.shape
    B_hat = np.random.randn(B_row, rank)
    C_hat = np.random.randn(C_row, rank)
    
    # Adding AWGN noise:
    if SNR_dB != None:
        X_noisy = fold(alphaV(unfold(X,1), SNR_dB), X.shape, 1) 
        X = X + X_noisy
    else:
        pass
    
    X_1 = unfold(X, 1)
    X_2 = unfold(X, 2)
    X_3 = unfold(X, 3)

    error = np.zeros([n_runs])
    error[0] = 0
    
    for i in range(1, n_runs):
        A_hat = X_1 @ np.linalg.pinv(kr(C_hat, B_hat).T)
        B_hat = X_2 @ np.linalg.pinv(kr(C_hat, A_hat).T)
        C_hat = X_3 @ np.linalg.pinv(kr(B_hat, A_hat).T)

        error[i] = np.linalg.norm(X_1 - A_hat @ (kr(C_hat, B_hat).T), 'fro')
        
        if abs(error[i] - error[i - 1]) <= delta:
            #print(f'Converged  with error {abs(error[i] - error[i - 1])}')
            break
    
    scaling_factor_a = known_values[0]/A_hat[0]
    scaling_factor_b = known_values[1]/B_hat[0]
    scaling_factor_c = known_values[2]/C_hat[0]
    
    A_hat_scaled = scaling_factor_a * A_hat
    B_hat_scaled = scaling_factor_b * B_hat
    C_hat_scaled = scaling_factor_c * C_hat
    
    return A_hat_scaled, B_hat_scaled, C_hat_scaled