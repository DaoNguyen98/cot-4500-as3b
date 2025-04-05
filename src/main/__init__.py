import numpy as np

# Solves a linear system using Gaussian Elimination
def gaussian_elimination():
    matrix = np.array([
        [2, -1, 1, 6],
        [1, 3, 1, 0],
        [-1, 5, 4, -3]
    ], dtype=float)
    
    size = len(matrix)

    # Forward elimination
    for pivot_row in range(size):
        # Swap with a row below if pivot is zero
        if matrix[pivot_row, pivot_row] == 0:
            for row in range(pivot_row + 1, size):
                if matrix[row, pivot_row] != 0:
                    matrix[[pivot_row, row]] = matrix[[row, pivot_row]]
                    break
        
        # Eliminate entries below the pivot
        for row in range(pivot_row + 1, size):
            factor = matrix[row, pivot_row] / matrix[pivot_row, pivot_row]
            matrix[row, pivot_row:] -= factor * matrix[pivot_row, pivot_row:]
    
    # Back substitution
    solution = np.zeros(size)
    for row in range(size - 1, -1, -1):
        solution[row] = (
            matrix[row, -1] - np.dot(matrix[row, row+1:size], solution[row+1:size])
        ) / matrix[row, row]
    
    print(solution, "\n")

# Performs LU Factorization and computes the determinant
def lu_factorization():
    matrix = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ], dtype=float)
    
    size = len(matrix)
    lower = np.eye(size)     # Identity matrix for L
    upper = np.zeros((size, size))  # Empty matrix for U

    # LU Decomposition (Doolittle's method)
    for i in range(size):
        for j in range(i, size):
            upper[i, j] = matrix[i, j] - np.dot(lower[i, :i], upper[:i, j])
        for j in range(i + 1, size):
            lower[j, i] = (matrix[j, i] - np.dot(lower[j, :i], upper[:i, i])) / upper[i, i]
    
    determinant = np.prod(np.diag(upper))  # Product of U's diagonal = det(A)

    print(determinant, "\n")
    print(lower, "\n")
    print(upper, "\n")

# Checks if the matrix is diagonally dominant
def is_diagonally_dominant():
    matrix = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    for i in range(len(matrix)):
        diagonal = abs(matrix[i, i])
        row_sum = sum(abs(matrix[i, :])) - diagonal
        if diagonal < row_sum:
            return False
    return True

# Checks if a matrix is symmetric
def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

# Extracts top-left k x k submatrix
def sub_matrix(matrix, k):
    return matrix[:k, :k]

# Returns the determinant of a matrix
def determinant(matrix):
    return np.linalg.det(matrix)

# Checks if a matrix is positive definite
def is_positive_definite():
    matrix = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    if not is_symmetric(matrix):
        return False
    for k in range(1, len(matrix) + 1):
        if determinant(sub_matrix(matrix, k)) <= 0:
            return False
    return True

if __name__ == "__main__":
    gaussian_elimination()
    lu_factorization()
    print(is_diagonally_dominant(), "\n")
    print(is_positive_definite())

