

import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib
import joblib


# Save the model as a pickle in a file

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
results = []

from point_cloud import *

# =========================
# Read point clouds
# =========================
selected_features = ['PCA1', 'sphericity', 'verticality', 'nx', 'nz', 'intensity']
point_cloud_df, my_point_cloud, offsets, scales = read_point_cloud(r"data/Original_Point_Cloud.las")
print(point_cloud_df.shape)
selected_features = ['PCA1', 'sphericity', 'verticality', 'nx', 'nz', 'intensity']

model = joblib.load('RandomForest.pkl')

X, feature_names = extract_feature(point_cloud_df, n_radius)
print("Extracted")
selected_feature_id = [feature_names.index(f) for f in selected_features]
y = model.predict(X[:,selected_feature_id])
def save_label(point_cloud, label, name):
  point_cloud.Class_Label =label
  point_cloud.write(f'{name}.las')
point_cloud = add_dimension(my_point_cloud, 'Class_Label', 'int')
save_label(point_cloud, y, 'Classified_Point_Cloud')

