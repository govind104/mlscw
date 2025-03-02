import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

# Choose the GPU if available; otherwise, fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def distance_cosine(x, y):
    """
    Compute cosine distance between two vectors
    d(X, Y) = 1 - (X · Y) / (||X|| ||Y||)
    
    Args:
        x (torch.Tensor): First vector of shape [D]
        y (torch.Tensor): Second vector of shape [D]
    
    Returns:
        torch.Tensor: Cosine distance (scalar)
    """
    dot_product = torch.dot(x, y)
    norm_x = torch.norm(x)
    norm_y = torch.norm(y)
    
    # Handle zero-norm vectors to avoid division by zero
    if norm_x.item() == 0 or norm_y.item() == 0:
        return torch.tensor(1.0, device = x.device)
    
    return 1.0 - dot_product / (norm_x * norm_y)


def distance_l2(x, y):
    """
    Compute L2 (Euclidean) distance between two vectors
    d(X, Y) = sqrt(sum_i (X_i - Y_i)^2)
    
    Args:
        x (torch.Tensor): First vector of shape [D]
        y (torch.Tensor): Second vector of shape [D]
    
    Returns:
        torch.Tensor: L2 distance (scalar)
    """
    return torch.sqrt(torch.sum((x - y) ** 2))


def distance_dot(x, y):
    """
    Compute dot product between two vectors
    d(X, Y) = X · Y
    
    Args:
        x (torch.Tensor): First vector of shape [D]
        y (torch.Tensor): Second vector of shape [D]
    
    Returns:
        torch.Tensor: Dot product (scalar)
    """
    return torch.dot(x, y)


def distance_manhattan(x, y):
    """
    Compute Manhattan (L1) distance between two vectors
    d(X, Y) = sum_i |X_i - Y_i|
    
    Args:
        x (torch.Tensor): First vector of shape [D]
        y (torch.Tensor): Second vector of shape [D]
    
    Returns:
        torch.Tensor: Manhattan distance (scalar)
    """
    return torch.sum(torch.abs(x - y))

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    """
    Find the top K nearest vectors to X from a collection A using L2 distance.
    
    Args:
      N (int): Number of vectors in A.
      D (int): Dimension of each vector.
      A (numpy.ndarray): Array of shape [N, D] containing the dataset.
      X (numpy.ndarray): Query vector of shape [D].
      K (int): Number of nearest neighbors to retrieve.
    
    Returns:
      numpy.ndarray: Top K nearest vectors (each of dimension D).
    """
    # Convert the dataset and query vector to torch tensors on the device.
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Compute the L2 distance for each vector in A (broadcasting X_tensor)
    distances = torch.norm(A_tensor - X_tensor, dim=1)
    
    # Get the indices of the K smallest distances
    indices = torch.argsort(distances)[:K]
    
    # Retrieve the top K vectors
    top_k_vectors = A_tensor[indices]
    
    # Return the result to CPU as a NumPy array
    return top_k_vectors.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
