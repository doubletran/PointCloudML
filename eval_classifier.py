

import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
results = []

from point_cloud import *

# =========================
# Read point clouds
# =========================
trees_df, _, _, _ = read_point_cloud(r"data\train\Trees.las")
ground_df, _, _, _ = read_point_cloud(r"data\train\Ground.las")
cars_df, _, _, _ = read_point_cloud(r"data\train\Cars.las")
buildings_df, _, _, _ = read_point_cloud(r"data\train\Buildings.las")


# =========================
# Extract features
# =========================
X_tree, feature_names = extract_feature(trees_df, n_radius)
X_ground, _ = extract_feature(ground_df, n_radius)
X_car, _ = extract_feature(cars_df, n_radius)
X_building, _ = extract_feature(buildings_df, n_radius)


# =========================
# Generate labels
# =========================
y_tree = np.ones(len(X_tree)) * 1        # trees
y_ground = np.ones(len(X_ground)) * 0    # ground
y_car = np.ones(len(X_car)) * 2          # cars
y_building = np.ones(len(X_building)) * 3 # buildings


# =========================
# Combine dataset
# =========================
X_train= np.vstack([X_tree,X_ground,X_car,X_building])

y_train= np.concatenate([y_tree,y_ground,y_car,y_building])
train_dataset = pd.DataFrame(
    np.hstack((X_train, y_train.reshape(-1,1))),
    columns = feature_names + ["label"]
)
train_dataset.to_csv("train_data.csv", index=False)
# =========================
# Read point clouds (TEST)
# =========================
trees_df_test, tree_point_cloud, _, _ = read_point_cloud(r"data\test\Trees.las")
ground_df_test, _, _, _ = read_point_cloud(r"data\test\Ground.las")
cars_df_test, car_point_cloud, _, _ = read_point_cloud(r"data\test\Cars.las")
buildings_df_test, _, _, _ = read_point_cloud(r"data\test\Buildings.las")


# =========================
# Extract features
# =========================
X_tree_test, _ = extract_feature(trees_df_test, n_radius)
X_ground_test, _ = extract_feature(ground_df_test, n_radius)
X_car_test, _ = extract_feature(cars_df_test, n_radius)
X_building_test, _ = extract_feature(buildings_df_test, n_radius)


# =========================
# Generate labels
# =========================
y_tree_test = np.ones(len(X_tree_test)) * 1        # trees
y_ground_test = np.ones(len(X_ground_test)) * 0    # ground
y_car_test = np.ones(len(X_car_test)) * 2          # cars
y_building_test = np.ones(len(X_building_test)) * 3 # buildings


# =========================
# Combine dataset
# =========================
X_test = np.vstack([X_tree_test,X_ground_test,X_car_test,X_building_test])
y_test = np.concatenate([y_tree_test,y_ground_test, y_car_test, y_building_test])
test_dataset = pd.DataFrame(
    np.hstack((X_test, y_test.reshape(-1,1))),
    columns = feature_names + ["label"]
)

test_dataset.to_csv("test_data.csv", index=False)


def evaluate_model(name, strategy, pipe, Xtr, ytr, Xte, yte):
    print("training")
    # train time
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_time = time.perf_counter() - t0
    print("testing")
    # test time
    t1 = time.perf_counter()
    y_pred = pipe.predict(Xte)
    test_time = time.perf_counter() - t1

    cm = confusion_matrix(yte, y_pred, labels=[0,1,2,3])
    acc = accuracy_score(yte, y_pred)
    kappa = cohen_kappa_score(yte, y_pred)
    f1m = f1_score(yte, y_pred, average="macro")

    # hyperparameters (even if default)
    params = pipe.get_params(deep=False)

    row = {
        "Strategy": strategy,
        "Classifier": name,
        "TrainTime_s": train_time,
        "TestTime_s": test_time,
        "Accuracy": acc,
        "Kappa": kappa,
        "F1_macro": f1m,
        "ConfusionMatrix": cm,
        "Hyperparams": params
    }
    return row, y_pred, pipe
classifiers = {
    "GaussianNB": GaussianNB(),
    "SVM_rbf": SVC(kernel="poly", C=1, gamma="scale"),          # arbitrary kernel (rbf)
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    ),
    "xgboost":XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softmax",
        num_class=4,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42
    )
}
def sfs(X, y, feature_names, k=6):
    print("Selecting features")
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    base_estimator =classifiers["RandomForest"]

    pipe_sfs = Pipeline([
        ("scaler", StandardScaler()),
        ("sfs", SequentialFeatureSelector(
            estimator=base_estimator,
            n_features_to_select=k,
            direction="forward",
            scoring="accuracy",
            cv=5,
            n_jobs=-1
        )),
        ("clf", base_estimator)
    ])

    # train pipeline
    row, y_pred, fitted_pipe = evaluate_model(
        "GaussianNB", f"SFS_forward(k=6)", pipe_sfs, X_train, y_train, X_test, y_test
    )
    print(row)
    # get selected features
    mask = fitted_pipe.named_steps["sfs"].get_support()
    selected_features = list(np.array(feature_names)[mask])

    return selected_features
def feature_model(X_train, y_train, X_test, y_test, selected_feature_idx, clf_name):
  #  SFS + classifier (Pipeline prevents leakage)
  pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", classifiers[clf_name])
  ])
  row, y_pred, fitted_pipe = evaluate_model(
      clf_name, f"SFS", pipe, X_train[:,selected_feature_idx], y_train, X_test[:, selected_feature_idx], y_test
  )
  print(row)
  return row, y_pred, fitted_pipe
def pca( X_train, y_train,X_test, y_test, clf_name="GaussianNB",):
  # 5) SFS + classifier (Pipeline prevents leakage)
  pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("pca", PCA(n_components=0.95, svd_solver="full")),
      ("clf", classifiers[clf_name])
  ])
  row, y_pred, fitted_pipe = evaluate_model(
      clf_name, f"PCA_95", pipe, X_train, y_train, X_test, y_test
  )
  print(row)
  return row, y_pred, fitted_pipe


#selected_features = sfs(X_train, y_train, feature_names)
def save_label(point_cloud, label, name):
  point_cloud.Class_Label =label
  point_cloud.write(f'{name}.las')
"""
selected_features = ['PCA1', 'sphericity', 'verticality', 'nx', 'nz', 'intensity']
#selected_features = ['eigenvalue_sum', 'verticality', 'nx', 'ny', 'intensity', 'scan_angle_rank']


selected_feature_idx = [feature_names.index(f) for f in selected_features]
row, y_pred, model = feature_model(X_train, y_train, X_test, y_test, selected_feature_idx, "xgboost")
from joblib import Parallel, delayed
import joblib


# Save the model as a pickle in a file
joblib.dump(model, 'xgboost.pkl')

_, y_pred_pca, model_pca= pca(X_train, y_train, X_test, y_test, "RandomForest")

#Workflow to classify tree with Random Forest
tree_pred = model.predict(X_tree[:, selected_feature_idx])
tree_pred_pca = model_pca.predict(X_tree)
tree_point_cloud = add_dimension(tree_point_cloud, 'Class_Label', 'int')
save_label(tree_point_cloud, tree_pred, 'Forest_SFS_Tree')
save_label(tree_point_cloud, tree_pred_pca, 'Forest_PCA_Tree')
"""
#Selected features by 