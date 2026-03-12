

import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score, f1_score
results = []

import pandas as pd
train_dataset = pd.read_csv("train_data.csv")
X_train = train_dataset.drop(columns=["label"]).values
y_train = train_dataset["label"].values
feature_names = train_dataset.drop(columns=["label"]).columns.tolist()

test_dataset = pd.read_csv("test_data.csv")
X_test = test_dataset.drop(columns=["label"]).values
y_test = test_dataset["label"].values



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

selected_features = ['PCA1', 'sphericity', 'verticality', 'nx', 'nz', 'intensity', "intensity", "return_number"]
#selected_features = ['eigenvalue_sum', 'verticality', 'nx', 'ny', 'intensity', 'scan_angle_rank']


selected_feature_idx = [feature_names.index(f) for f in selected_features]

from joblib import Parallel, delayed
import joblib

row, y_pred, model = feature_model(X_train, y_train, X_test, y_test, selected_feature_idx, "RandomForest")
# Save the model as a pickle in a file
joblib.dump(model, 'model/RandomForestSFS.pkl')

"""
_, y_pred_pca, model_pca= pca(X_train, y_train, X_test, y_test, "RandomForest")

#Workflow to classify tree with Random Forest
tree_pred = model.predict(X_tree[:, selected_feature_idx])
tree_pred_pca = model_pca.predict(X_tree)
tree_point_cloud = add_dimension(tree_point_cloud, 'Class_Label', 'int')
save_label(tree_point_cloud, tree_pred, 'Forest_SFS_Tree')
save_label(tree_point_cloud, tree_pred_pca, 'Forest_PCA_Tree')
"""
#Selected features by 