import laspy
import pandas as pd
import numpy as np
from jakteristics import compute_features
# load the classification function
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# for n__radius in [0.5, 1, 1.2, 3]
n_radius = 1

# Add Dimension Function
def add_dimension(las_file, dimension_name, dtype):
    """  This function is used to add new dimension to the point cloud """ 
    if dtype == 'float':        
        las_file.add_extra_dim(laspy.ExtraBytesParams(
            name=dimension_name,
            type=np.float64,
            description=dimension_name
        )) 
    elif dtype == 'int':
        las_file.add_extra_dim(laspy.ExtraBytesParams(
            name=dimension_name,
            type=np.int32,
            description=dimension_name
        ))                     
    return las_file

# Normalization Function
def normalize_feature(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val) if max_val > min_val else arr * 0


# Read Point Cloud function
def read_point_cloud(las_path):
    my_point_cloud = laspy.read(las_path)
    offsets = my_point_cloud.header.offsets
    scales = my_point_cloud.header.scales
    x = my_point_cloud['X']*scales[0] + offsets[0]
    y = my_point_cloud['Y']*scales[1] + offsets[1]
    z = my_point_cloud['Z']*scales[2] + offsets[2]
    #######################
    point_cloud_data_frame = pd.DataFrame()
    point_cloud_data_frame['x']  = x
    point_cloud_data_frame['y']  = y
    point_cloud_data_frame['z']  = z
    dimension_names = list(my_point_cloud.point_format.dimension_names)
    for i, dimension in enumerate(dimension_names):
        point_cloud_data_frame[dimension] = np.array(getattr(my_point_cloud, dimension))
    return point_cloud_data_frame, my_point_cloud, offsets, scales
import numpy as np

# Sequential Feature Extractor (feature builder)
def extract_feature(point_cloud_df, n_radius, add_extra=("intensity", "return_number")):
    selected_geometric_features = [
        "eigenvalue_sum", "omnivariance", "eigenentropy", "anisotropy", "planarity", "linearity",
        "PCA1", "PCA2", "surface_variation", "sphericity", "verticality",
        "nx", "ny", "nz", "number_of_neighbors"
    ]
    selected_geometric_features =  ['PCA1', 'sphericity', 'verticality', 'nx', 'nz']

    # xyz as numpy
    xyz = point_cloud_df[["x", "y", "z"]].to_numpy(dtype=float)

    # compute geometric features: (N, G)
    geometric_features = compute_features(
        xyz,
        search_radius=n_radius,
        feature_names=selected_geometric_features
    )
    geometric_features = np.asarray(geometric_features, dtype=float)
    geometric_features[np.isnan(geometric_features)] = 0.0

    # build output feature list
    feature_names = list(selected_geometric_features)

    extra_arrays = []
    for col in add_extra:
        if col in point_cloud_df.columns:
            arr = point_cloud_df[col].to_numpy(dtype=float).reshape(-1, 1)
            arr[np.isnan(arr)] = 0.0
            extra_arrays.append(arr)
            feature_names.append(col)

    # final X
    if extra_arrays:
        X = np.hstack([geometric_features] + extra_arrays)
    else:
        X = geometric_features

    return X, feature_names
