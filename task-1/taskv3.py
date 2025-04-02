import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import adjusted_rand_score

# ------------------------------------------------------------------------------------------------
# Distance functions using CuPy
# ------------------------------------------------------------------------------------------------

def distance_cosine(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    dot_product = cp.dot(X_cp, Y_cp)
    norm_X = cp.linalg.norm(X_cp)
    norm_Y = cp.linalg.norm(Y_cp)
    if norm_X == 0 or norm_Y == 0:
        return 1.0
    cosine_sim = dot_product / (norm_X * norm_Y)
    return (1 - cosine_sim).item()


def distance_l2(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    diff = X_cp - Y_cp
    return cp.sum(diff ** 2).item()


def distance_dot(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    return cp.dot(X_cp, Y_cp).item()


def distance_manhattan(X, Y):
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    diff = X_cp - Y_cp
    return cp.sum(cp.abs(diff)).item()

# ------------------------------------------------------------------------------------------------
# K-Nearest Neighbors using CuPy
# ------------------------------------------------------------------------------------------------

def our_knn(N, D, A, X, K):
    A_cp = cp.asarray(A, dtype=cp.float32)
    X_cp = cp.asarray(X, dtype=cp.float32)
    
    differences = A_cp - X_cp[None, :]
    squared_distances = cp.sum(differences ** 2, axis=1)
    indices = cp.argsort(squared_distances)[:K]
    
    return indices.get()

# ------------------------------------------------------------------------------------------------
# K-Means Clustering using CuPy
# ------------------------------------------------------------------------------------------------

# def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4, return_centroids=False):
#     data = cp.asarray(A, dtype=cp.float32)
    
#     indices = cp.random.permutation(N)[:K]
#     centroids = data[indices].copy()
#     cluster_ids = cp.zeros(N, dtype=cp.int32)
    
#     for _ in range(max_iters):
#         dists = cp.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
#         new_cluster_ids = cp.argmin(dists, axis=1)
        
#         if cp.all(new_cluster_ids == cluster_ids):
#             break
#         cluster_ids = new_cluster_ids
        
#         for k in range(K):
#             mask = cluster_ids == k
#             if cp.any(mask):
#                 centroids[k] = cp.mean(data[mask], axis=0)
#             else:
#                 centroids[k] = data[cp.random.randint(N)]
    
#     if return_centroids:
#         return cluster_ids.get(), centroids.get()
#     else:
#         return cluster_ids.get()

def our_kmeans(N, D, A, K, max_iters=100, tol=1e-4, return_centroids=False):
    data = cp.asarray(A, dtype=cp.float32)  # Ensure data is a CuPy array
    
    indices = cp.random.permutation(N)[:K]
    centroids = data[indices].copy()
    cluster_ids = cp.zeros(N, dtype=cp.int32)
    
    for _ in range(max_iters):
        dists = cp.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        new_cluster_ids = cp.argmin(dists, axis=1)
        
        if cp.all(new_cluster_ids == cluster_ids):
            break
        cluster_ids = new_cluster_ids
        
        for k in range(K):
            mask = cluster_ids == k
            if cp.any(mask):
                centroids[k] = cp.mean(data[mask], axis=0)
            else:
                centroids[k] = data[cp.random.randint(N)]
    
    if return_centroids:
        return cluster_ids.get(), centroids.get()  # Ensure centroids are returned as NumPy arrays
    else:
        return cluster_ids  # Return as CuPy array

# ------------------------------------------------------------------------------------------------
# Approximate Nearest Neighbors using CuPy
# ------------------------------------------------------------------------------------------------

# def our_ann(N, D, A, X, K):
#     A_cp = cp.asarray(A, dtype=cp.float32)
#     X_cp = cp.asarray(X, dtype=cp.float32)
    
#     cluster_num = 50
#     K1 = 10
#     K2 = 25
    
#     clusters = our_kmeans(N, D, A, cluster_num)
#     centroids = cp.zeros((cluster_num, D), dtype=cp.float32)
    
#     for i in range(cluster_num):
#         indices = cp.where(clusters == i)[0]
#         if indices.size > 0:
#             centroids[i] = cp.mean(A_cp[indices], axis=0)
#         else:
#             centroids[i] = cp.zeros(D, dtype=cp.float32)
    
#     centroid_dists = cp.array([distance_l2(X_cp, centroids[i]) for i in range(cluster_num)])
#     nearest_cluster_indices = cp.argsort(centroid_dists)[:K1]
    
#     candidate_indices_list = []
#     for c in nearest_cluster_indices:
#         indices = cp.where(clusters == c)[0]
#         if indices.size == 0:
#             continue
#         points_in_cluster = A_cp[indices]
#         dists_in_cluster = cp.array([distance_l2(X_cp, points_in_cluster[i]) for i in range(indices.size)])
#         k2 = min(K2, indices.size)
#         top_k_in_cluster = cp.argsort(dists_in_cluster)[:k2]
#         candidate_indices_list.append(indices[top_k_in_cluster])
    
#     if not candidate_indices_list:
#         return cp.array([])
    
#     candidate_indices = cp.concatenate(candidate_indices_list)
#     candidate_points = A_cp[candidate_indices]
#     candidate_dists = cp.array([distance_l2(X_cp, candidate_points[i]) for i in range(candidate_indices.size)])
#     k_final = min(K, candidate_dists.size)
#     final_top_k = cp.argsort(candidate_dists)[:k_final]
#     ann_indices = candidate_indices[final_top_k]
    
#     return ann_indices.get()

def our_ann(N, D, A, X, K):
    A_cp = cp.asarray(A, dtype=cp.float32)  # Ensure A is a CuPy array
    X_cp = cp.asarray(X, dtype=cp.float32)  # Ensure X is a CuPy array
    
    cluster_num = 50
    K1 = 10
    K2 = 25
    
    clusters = our_kmeans(N, D, A_cp, cluster_num)  # Ensure clusters is a CuPy array
    centroids = cp.zeros((cluster_num, D), dtype=cp.float32)
    
    for i in range(cluster_num):
        indices = cp.where(clusters == i)[0]  # This should now work as clusters is a CuPy array
        if indices.size > 0:
            centroids[i] = cp.mean(A_cp[indices], axis=0)
        else:
            centroids[i] = cp.zeros(D, dtype=cp.float32)
    
    centroid_dists = cp.array([distance_l2(X_cp, centroids[i]) for i in range(cluster_num)])
    nearest_cluster_indices = cp.argsort(centroid_dists)[:K1]
    
    candidate_indices_list = []
    for c in nearest_cluster_indices:
        indices = cp.where(clusters == c)[0]  # Ensure this is a CuPy operation
        if indices.size == 0:
            continue
        points_in_cluster = A_cp[indices]
        dists_in_cluster = cp.array([distance_l2(X_cp, points_in_cluster[i]) for i in range(indices.size)])
        k2 = min(K2, indices.size)
        top_k_in_cluster = cp.argsort(dists_in_cluster)[:k2]
        candidate_indices_list.append(indices[top_k_in_cluster])
    
    if not candidate_indices_list:
        return cp.array([])
    
    candidate_indices = cp.concatenate(candidate_indices_list)
    candidate_points = A_cp[candidate_indices]
    candidate_dists = cp.array([distance_l2(X_cp, candidate_points[i]) for i in range(candidate_indices.size)])
    k_final = min(K, candidate_dists.size)
    final_top_k = cp.argsort(candidate_dists)[:k_final]
    ann_indices = candidate_indices[final_top_k]
    
    return ann_indices.get()

# ------------------------------------------------------------------------------------------------
# Evaluation Functions
# ------------------------------------------------------------------------------------------------

# def recall_rate(true_indices, pred_indices, K=10):
#     return len(set(true_indices[:K]) & set(pred_indices[:K])) / K

# def recall_rate(list1, list2):
#     """
#     Calculate the recall rate of two lists
#     list1[K]: The top K nearest vectors ID
#     list2[K]: The top K nearest vectors ID
#     """
#     return len(set(list1) & set(list2)) / len(set(list1))


def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    # if list1 is None :
    #     raise ValueError("Input list 1 cannot be None.")
    # if list2 is None:
    #     raise ValueError("Input list 2 cannot be None.")
    
    return len(set(list1) & set(list2)) / len(list1) if len(list1) > 0 else 0.0

def benchmark_large_scale(test_file=""):
    N, D, A, X, K = testdata_ann(test_file)
    start_time = time.time()
    indices = our_ann(N, D, A, X, K)
    elapsed_time = time.time() - start_time
    print(f"ANN Execution Time: {elapsed_time:.4f} seconds")
    print("ANN Results:", indices)

def test_knn(test_file=""):
    N, D, A, X, K = testdata_knn(test_file)
    indices = our_knn(N, D, A, X, K)
    print("KNN Results:", indices)
    return indices

def test_kmeans(test_file=""):
    N, D, A, K = testdata_kmeans(test_file)
    cluster_ids = our_kmeans(N, D, A, K)
    print("K-Means Results:", cluster_ids)
    # print(cluster_ids.dtype)
    return cluster_ids

def test_ann(test_file=""):
    N, D, A, X, K = testdata_ann(test_file)
    indices = our_ann(N, D, A, X, K)
    print("ANN Results:", indices)
    return indices

if __name__ == "__main__":
    print("Testing K-Means...")
    test_kmeans()
    print("Testing KNN...")
    test_knn()
    print("Testing ANN...")
    test_ann()
    print("Benchmarking Large Scale ANN...")
    benchmark_large_scale()
    print("All tests completed.")

    l_knn = test_knn()
    l_ann = test_ann()
    print(f'Recall Rate: {recall_rate(l_knn, l_ann)}')
