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
    # Convert A and X to tensors on GPU
    A_tensor = torch.tensor(A, dtype=torch.float32, device='cuda')  # (N, D)
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')  # (D,)
    
    # Compute distances using predefined distance_l2 (inefficient due to loops)
    distances = []
    for i in range(N):
        # Convert tensors to numpy for the predefined function
        a_np = A_tensor[i].cpu().numpy()
        x_np = X_tensor.cpu().numpy()
        dist = distance_l2(a_np, x_np)
        distances.append(dist)
    distances = torch.tensor(distances, device='cuda')  # (N,)
    
    # Avoid topk: use argsort + slicing
    sorted_indices = torch.argsort(distances)
    indices = sorted_indices[:K]

    return indices.cpu().numpy()

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4, return_centroids=False, distance_metric='l2'):
    data = torch.tensor(A, dtype=torch.float32, device='cuda')  # (N, D)
    
    # Initialize centroids
    indices = torch.randperm(N, device='cuda')[:K]
    centroids = data[indices].clone()  # (K, D)
    
    cluster_ids = torch.zeros(N, dtype=torch.long, device='cuda')
    
    for _ in range(max_iters):
        # Compute distances using predefined functions (nested loops)
        dists = torch.zeros((N, K), device='cuda')
        for i in range(N):
            for k in range(K):
                data_point = data[i].cpu().numpy()  # Inefficient!
                centroid = centroids[k].cpu().numpy()
                if distance_metric == 'cosine':
                    dist = distance_cosine(data_point, centroid)
                else:
                    dist = distance_l2(data_point, centroid)
                dists[i, k] = dist
        
        new_cluster_ids = torch.argmin(dists, dim=1)
        
        if torch.all(new_cluster_ids == cluster_ids):
            break
        cluster_ids = new_cluster_ids
        
        # Update centroids (unchanged)
        new_centroids = torch.zeros_like(centroids)
        for k in range(K):
            mask = cluster_ids == k
            if mask.any():
                new_centroids[k] = data[mask].mean(dim=0)
            else:
                new_centroids[k] = data[torch.randint(N, (1,), device='cuda')]
        centroids = new_centroids
    
    return cluster_ids.cpu().numpy() if not return_centroids else (cluster_ids, centroids)

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

def our_ann(N, D, A, X, K, ann_clusters=100, search_clusters=5, distance_metric='l2'):
    A_tensor = torch.tensor(A, dtype=torch.float32, device='cuda')
    X_tensor = torch.tensor(X, dtype=torch.float32, device='cuda')
    
    # Step 1: Cluster data (using predefined distance functions)
    cluster_ids, centroids = our_kmeans(N, D, A, ann_clusters, return_centroids=True, distance_metric=distance_metric)
    
    # Step 2: Find nearest clusters to query
    cluster_distances = []
    for k in range(ann_clusters):
        centroid_np = centroids[k].cpu().numpy()
        X_np = X_tensor.cpu().numpy()
        if distance_metric == 'cosine':
            dist = distance_cosine(X_np, centroid_np)
        else:
            dist = distance_l2(X_np, centroid_np)
        cluster_distances.append(dist)
    cluster_distances = torch.tensor(cluster_distances, device='cuda')
    
    # Avoid topk for cluster selection
    sorted_cluster_indices = torch.argsort(cluster_distances)
    closest_clusters = sorted_cluster_indices[:search_clusters]
    
    # Step 3: Collect candidates from clusters
    candidate_indices = []
    for c in closest_clusters:
        mask = (cluster_ids == c).cpu().numpy()
        candidates = A[mask]  # A is a numpy array
        for cand in candidates:
            if distance_metric == 'cosine':
                dist = distance_cosine(X, cand)
            else:
                dist = distance_l2(X, cand)
            candidate_indices.append((dist, np.where(mask)[0][0]))
    
    # Step 4: Sort candidates and return top-K
    candidate_indices.sort(key=lambda x: x[0])
    final_indices = [idx for _, idx in candidate_indices[:K]]
    return np.array(final_indices)

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

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
