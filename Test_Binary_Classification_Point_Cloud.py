import laspy
import pandas as pd
import numpy as np
from jakteristics import compute_features
# load the classification function
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SequentialFeatureSelector
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

point_cloud_data_frame, my_point_cloud, offsets, scales = read_point_cloud(r"Small_Area_point_cloud.las")
dimension_names = print(list)
# feature extraction
''' Potential Features
list of selected geometric features to be extracted among the following list:

'''
geometric_features_names = [
    "eigenvalue_sum", "omnivariance", "eigenentropy", "anisotropy", "planarity", "linearity",
    "PCA1", "PCA2", "surface_variation", "sphericity", "verticality",
    "nx", "ny", "nz", "number_of_neighbors",
    "eigenvalue1", "eigenvalue2", "eigenvalue3",
    "eigenvector1x", "eigenvector1y", "eigenvector1z",
    "eigenvector2x", "eigenvector2y", "eigenvector2z",
    "eigenvector3x", "eigenvector3y", "eigenvector3z"
]
selected_geometric_features = ["planarity", "linearity", "PCA1", "PCA", "surface_variation", "sphericity", "eigenvalue1", "nz", "anisotropy"]
xyz = point_cloud_data_frame[["x","y","z"]]
geometric_features = compute_features(xyz, search_radius=n_radius, feature_names=selected_geometric_features)
nan_values = np.isnan(geometric_features)
geometric_features[nan_values] = 0

for i, feature_name in enumerate(selected_geometric_features):
    point_cloud_data_frame[feature_name] = geometric_features[:,i]


selected_features_for_classification = ["intensity", "return_number", "planarity", "linearity","PCA1", "PCA2", "anisotropy", "nz"]
calssification_features = point_cloud_data_frame[selected_features_for_classification]

# Normalization
for dim in selected_features_for_classification:
    calssification_features[dim] = normalize_feature(calssification_features[dim])


# read training samples
trees_las_path = r"Training_Data\Trees.las"
Ground_las_path = r"Training_Data\Ground.las"

trees_points, _, _, _ = read_point_cloud(trees_las_path)
tree_ids = trees_points['Id']
ground_points, _, _, _ = read_point_cloud(Ground_las_path)
ground_ids = ground_points['Id']


Sample_data_tree = calssification_features[point_cloud_data_frame['Id'].isin(tree_ids)]
Sample_data_ground = calssification_features[point_cloud_data_frame['Id'].isin(ground_ids)]

sample_features_both = np.concatenate([Sample_data_tree, Sample_data_ground])

# generate class labels for training data
class_labels_both = np.concatenate([np.ones(len(Sample_data_tree)), np.zeros(len(Sample_data_ground))])

# split test and train
from sklearn.model_selection import train_test_split
test_samples_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(sample_features_both, class_labels_both, test_size=test_samples_size)


'''
In the class we didn't talk about PCA or sequential feature selection'
You may use PCA on the train Data here. For the test and all data, you may use the fitted PCA to training data on test and all data before prediction.
'''

clf = GaussianNB()
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=2, random_state=0)
sfs = SequentialFeatureSelector(clf, n_features_to_select=3, direction="forward")
SequentialFeatureSelector(estimator=GaussianNB(n_neighbors=3),
                          n_features_to_select=3)
sfs_forward = sfs.fit(X_train, y_train)
print(
    "Features selected by forward sequential selection: "
    f"{geometric_features_names[sfs_forward.get_support()]}"
)
# fit the classification funciton to the data
clf.fit(X_train, y_train)

# predict test
y_test_poredict = clf.predict(X_test)

# Evaluate the resutls
from sklearn.metrics import confusion_matrix, f1_score
print(confusion_matrix(y_test, y_test_poredict))
print(f1_score(y_test, y_test_poredict))
all_labels = clf.predict(calssification_features)

# save the class labels as a scaler field of the point clouds
my_point_cloud = add_dimension(my_point_cloud, 'Class_Label', 'int')
my_point_cloud.Class_Label = all_labels
my_point_cloud.write('Classified_point_cloud.las')


