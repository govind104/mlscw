import torch

def cosine_distance(x, y):
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
    if norm_x == 0 or norm_y == 0:
        return torch.tensor(1.0, device=x.device)
    
    return 1.0 - dot_product / (norm_x * norm_y)


def l2_distance(x, y):
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


def dot_product(x, y):
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


def manhattan_distance(x, y):
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


# Vectorized versions for computing distances between one vector and many vectors
def cosine_distance_batch(x, y_batch):
    """
    Compute cosine distance between a vector and a batch of vectors
    
    Args:
        x (torch.Tensor): Vector of shape [D]
        y_batch (torch.Tensor): Batch of vectors of shape [N, D]
    
    Returns:
        torch.Tensor: Cosine distances of shape [N]
    """
    # Reshape x to [1, D] for broadcasting
    x = x.view(1, -1)
    
    # Compute dot products
    dot_products = torch.matmul(y_batch, x.t()).squeeze()
    
    # Compute norms
    norm_x = torch.norm(x)
    norm_y_batch = torch.norm(y_batch, dim=1)
    
    # Handle zero-norm vectors
    zero_mask = (norm_x == 0) | (norm_y_batch == 0)
    
    # Compute cosine distances
    similarities = dot_products / (norm_x * norm_y_batch)
    distances = 1.0 - similarities
    
    # Set distance to 1.0 for zero-norm vectors
    distances = torch.where(zero_mask, torch.ones_like(distances), distances)
    
    return distances


def l2_distance_batch(x, y_batch):
    """
    Compute L2 distances between a vector and a batch of vectors
    
    Args:
        x (torch.Tensor): Vector of shape [D]
        y_batch (torch.Tensor): Batch of vectors of shape [N, D]
    
    Returns:
        torch.Tensor: L2 distances of shape [N]
    """
    # Reshape x to [1, D] for broadcasting
    x = x.view(1, -1)
    
    # Compute squared differences
    squared_diff = (y_batch - x) ** 2
    
    # Sum across dimensions and take square root
    return torch.sqrt(torch.sum(squared_diff, dim=1))


def dot_product_batch(x, y_batch):
    """
    Compute dot products between a vector and a batch of vectors
    
    Args:
        x (torch.Tensor): Vector of shape [D]
        y_batch (torch.Tensor): Batch of vectors of shape [N, D]
    
    Returns:
        torch.Tensor: Dot products of shape [N]
    """
    # Use matrix multiplication for efficient computation
    return torch.matmul(y_batch, x)


def manhattan_distance_batch(x, y_batch):
    """
    Compute Manhattan distances between a vector and a batch of vectors
    
    Args:
        x (torch.Tensor): Vector of shape [D]
        y_batch (torch.Tensor): Batch of vectors of shape [N, D]
    
    Returns:
        torch.Tensor: Manhattan distances of shape [N]
    """
    # Reshape x to [1, D] for broadcasting
    x = x.view(1, -1)
    
    # Compute absolute differences and sum
    return torch.sum(torch.abs(y_batch - x), dim=1)