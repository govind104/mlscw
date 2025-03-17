import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import adjusted_rand_score

# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device='cuda')
    dot_product = torch.dot(X_tensor, Y_tensor)
    norm_X = torch.norm(X_tensor)
    norm_Y = torch.norm(Y_tensor)
    if norm_X == 0 or norm_Y == 0:
        return 1.0
    cosine_sim = dot_product / (norm_X * norm_Y)
    return (1 - cosine_sim).item()


def distance_l2(X, Y):
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device='cuda')
    diff = X_tensor - Y_tensor
    return torch.sum(diff ** 2).item()

def distance_dot(X, Y):
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device='cuda')
    return torch.dot(X_tensor, Y_tensor).item()

def distance_manhattan(X, Y):
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device='cuda')
    diff = X_tensor - Y_tensor
    return torch.sum(torch.abs(diff)).item()

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K):
    # Convert input arrays to PyTorch tensors on the GPU
    A_tensor = torch.tensor(A, dtype=torch.float32, device='cuda')
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    
    # Compute L2 squared distances between X and all vectors in A
    differences = A_tensor - X_tensor.unsqueeze(0)  # Broadcast subtraction
    squared_distances = torch.sum(differences ** 2, dim=1)
    
    # Get indices of K smallest distances
    _, indices = torch.topk(squared_distances, K, largest=False)
    
    # Return indices as numpy array (moved to CPU)
    return indices.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4, return_centroids=False):
    # Convert data to GPU tensor
    data = torch.tensor(A, dtype=torch.float32, device='cuda')
    
    # Initialize centroids randomly from data points
    indices = torch.randperm(N, device='cuda')[:K]
    centroids = data[indices].clone()
    
    cluster_ids = torch.zeros(N, dtype=torch.long, device='cuda')
    
    for _ in range(max_iters):
        # Compute pairwise distances and assign clusters
        dists = torch.cdist(data, centroids, p=2)
        new_cluster_ids = torch.argmin(dists, dim=1)
        
        # Check for convergence
        if torch.all(new_cluster_ids == cluster_ids):
            break
        cluster_ids = new_cluster_ids
        
        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = cluster_ids == k
            if mask.any():
                new_centroids[k] = data[mask].mean(dim=0)
            else:  # Reinitialize empty clusters
                new_centroids[k] = data[torch.randint(N, (1,), device='cuda')]
        centroids = new_centroids
    
    # Return based on the flag
    if return_centroids:
        return cluster_ids, centroids  # GPU tensors
    else:
        return cluster_ids.cpu().numpy()  # CPU numpy array

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K, ann_clusters=100, search_clusters=5):
    # Convert data to GPU tensors
    A_tensor = torch.tensor(A, dtype=torch.float32, device='cuda')
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    
    # Step 1: Cluster data using K-means (returns GPU tensors)
    cluster_ids, centroids = our_kmeans(N, D, A, ann_clusters, return_centroids=True)
    
    # Step 2: Find nearest clusters to the query
    query_dist = torch.cdist(X_tensor.unsqueeze(0), centroids).squeeze(0)
    _, closest_clusters = torch.topk(query_dist, search_clusters, largest=False)
    
    # Step 3: Collect candidate vectors from target clusters
    mask = torch.isin(cluster_ids, closest_clusters)
    candidates = A_tensor[mask]
    candidate_indices = torch.nonzero(mask).squeeze()
    
    # Step 4: Exact search within candidates
    candidate_dists = torch.cdist(X_tensor.unsqueeze(0), candidates).squeeze(0)
    _, topk_local = torch.topk(candidate_dists, K, largest=False)
    
    # Map back to original indices
    result = candidate_indices[topk_local].cpu().numpy()
    return result
# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    # Load test data
    N, D, A, K = testdata_kmeans("")
    
    # Test our implementation
    start = time.time()
    our_clusters = our_kmeans(N, D, A, K)
    gpu_time = time.time() - start
    
    # Compare with sklearn (CPU baseline)
    start = time.time()
    sklearn_kmeans = SKLearnKMeans(n_clusters=K).fit(A)
    cpu_time = time.time() - start
    sklearn_clusters = sklearn_kmeans.labels_
    
    # Calculate cluster similarity
    ari = adjusted_rand_score(sklearn_clusters, our_clusters)
    
    print(f"\nKMeans Test Results:")
    print(f"- GPU Time: {gpu_time:.4f}s | CPU Time: {cpu_time:.4f}s | Speedup: {cpu_time/gpu_time:.1f}x")
    print(f"- Adjusted Rand Index vs sklearn: {ari:.4f}")
    print(f"- Cluster distribution: {np.bincount(our_clusters)}")

def test_knn():
    # Load test data
    N, D, A, X, K = testdata_knn("")
    
    # GPU implementation
    start = time.time()
    our_result = our_knn(N, D, A, X, K)
    gpu_time = time.time() - start
    
    # CPU reference (numpy)
    start = time.time()
    differences = A - X.reshape(1, -1)
    distances = np.linalg.norm(differences, axis=1)
    ref_result = np.argpartition(distances, K)[:K]
    cpu_time = time.time() - start
    
    # Recall check
    recall = len(set(our_result) & set(ref_result)) / K
    
    print(f"\nKNN Test Results:")
    print(f"- GPU Time: {gpu_time:.4f}s | CPU Time: {cpu_time:.4f}s | Speedup: {cpu_time/gpu_time:.1f}x")
    print(f"- Recall vs CPU reference: {recall:.1%}")
    print(f"- Top {K} indices: {our_result}")

def test_ann():
    # Load test data
    N, D, A, X, K = testdata_ann("")
    
    # Exact KNN reference
    knn_result = our_knn(N, D, A, X, K)
    
    # ANN implementation
    start = time.time()
    ann_result = our_ann(N, D, A, X, K)
    ann_time = time.time() - start
    
    # Recall rate calculation
    recall = len(set(ann_result) & set(knn_result)) / K
    
    print(f"\nANN Test Results:")
    print(f"- ANN Time: {ann_time:.4f}s | KNN Time: See previous test")
    print(f"- Recall Rate: {recall:.1%} {'(PASS)' if recall >= 0.7 else '(FAIL)'}")
    print(f"- ANN Indices: {ann_result}")

def benchmark_large_scale():
    """Additional benchmark for 4M vectors scenario"""
    print("\nRunning Large-scale Benchmark (4000000 vectors)...")
    
    # Generate random data
    N, D = 4000000, 1024
    A = np.random.randn(N, D).astype(np.float32)
    X = np.random.randn(D).astype(np.float32)
    K = 10
    
    # GPU KNN
    start = time.time()
    our_knn(N, D, A, X, K)
    gpu_time = time.time() - start
    
    print(f"- GPU KNN Time for 4M vectors: {gpu_time:.1f}s")
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run all tests with recall checks
    print("="*40 + "\nRunning KMeans Test\n" + "="*40)
    test_kmeans()
    
    print("\n" + "="*40 + "\nRunning KNN Test\n" + "="*40)
    test_knn()
    
    print("\n" + "="*40 + "\nRunning ANN Test\n" + "="*40)
    test_ann()
    
    # Additional validation
    print("\n" + "="*40 + "\nCross-validation\n" + "="*40)
    N, D, A, X, K = testdata_ann("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    ann_result = our_ann(N, D, A, X, K)
    print(f"Final Recall Rate: {recall_rate(knn_result, ann_result):.1%}")
    
    # Uncomment for large-scale benchmark
    # benchmark_large_scale()