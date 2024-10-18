import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

# Sample data points
p1, p2, p3, p4, p5, p6, p7, p8 = (2,10), (2,5), (8,4), (5,8), (7,5), (6,4), (1,2), (4,9)
data_points = [p1, p2, p3, p4, p5, p6, p7, p8]

def calculate_euclidean_distance(point: Tuple, reference: Tuple) -> float:
    return np.sqrt(sum((p - r) ** 2 for p, r in zip(point, reference)))

def calculate_centroid(points: List[Tuple]) -> Tuple:
    """Calculate the centroid of a group of points"""
    if not points:
        return (None, None)
    return tuple(np.mean(points, axis=0))

def perform_clustering(data_points: List[Tuple], 
                      reference_points: List[Tuple], 
                      iteration: int = 0, 
                      max_iterations: int = 10) -> pd.DataFrame:
    """
    Recursively perform clustering until convergence or max iterations reached
    """
    # Create DataFrame for current iteration
    cluster_df = pd.DataFrame({"Data Points": data_points})
    
    # Calculate distances to each reference point
    for i, ref_point in enumerate(reference_points, 1):
        col_name = f"P{i}"
        cluster_df[col_name] = cluster_df['Data Points'].apply(
            lambda x: calculate_euclidean_distance(x, ref_point)
        )
    
    # Assign clusters
    cluster_df["Cluster"] = cluster_df[[f"P{i}" for i in range(1, len(reference_points)+1)]].idxmin(axis=1)
    
    # Calculate new centroids
    new_reference_points = []
    for i in range(1, len(reference_points)+1):
        cluster_points = cluster_df[cluster_df["Cluster"] == f"P{i}"]["Data Points"].tolist()
        centroid = calculate_centroid(cluster_points) if cluster_points else reference_points[i-1]
        new_reference_points.append(centroid)
    
    # Check for convergence or max iterations
    if iteration >= max_iterations or new_reference_points == reference_points:
        return cluster_df
    
    # Recursive call with new reference points
    print(f"Iteration {iteration + 1}:")
    print(f"New reference points: {new_reference_points}")
    return perform_clustering(data_points, new_reference_points, iteration + 1, max_iterations)

# Initial reference points
initial_reference_points = [p1, p4, p7]

# Perform recursive clustering
#result = perform_clustering(data_points, initial_reference_points)
#print("\nFinal Result:")
#print(result)


#-----------------------------------------------------------------------------
#K-Medions

a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 = (2,6), (3,8), (4,7), (6,2), (6,4), (7,3), (7,4), (8,5), (7,6), (3,4)
data_points = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]

def calculate_manhattan_distance(point: Tuple, reference: Tuple) -> float:
    return sum(abs(p-r) for p,r in zip(point, reference))

def perform_K_Med(data_points: List[Tuple], 
                      reference_points: List[Tuple]) -> pd.DataFrame:
    mediods_df = pd.DataFrame({"Data Points": data_points})

    for i, ref_point in enumerate(reference_points, 1):
        col_name = f"M{i}"
        mediods_df[col_name] = mediods_df['Data Points'].apply(
            lambda x: calculate_manhattan_distance(x, ref_point)
            )
    
    mediods_df["Cluster"] = mediods_df[[f"M{i}" for i  in range(1, len(reference_points)+1)]].idxmin(axis=1)

    for i in range(1, len(reference_points)+1):
        med_points = mediods_df[mediods_df["Cluster"] == f"M{i}"][f"M{i}"].tolist()
        print(f"M{i}: ", sum(med_points))

    return(mediods_df)

result = perform_K_Med(data_points, [a10, a6])
print(result)
result = perform_K_Med(data_points, [a10, a7])
print(result)
result = perform_K_Med(data_points, [a10, a5])
print(result)
