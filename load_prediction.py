import pandas as pd
from point_cloud import *
label=["GaussianNB_SFS_Pred","GaussianNB_PCA_Pred","RandomForest_SFS_Pred","RandomForest_PCA_Pred","xgboost_SFS_Pred","xgboost_PCA_Pred"]
all_predictions_df = pd.read_csv("test_data_with_predictions.csv")
# =========================
# Read point clouds (TEST)
# =========================
trees_df_test, tree_point_cloud, _, _ = read_point_cloud(r"data\test\Trees.las")
ground_df_test, ground_point_cloud, _, _ = read_point_cloud(r"data\test\Ground.las")
cars_df_test, car_point_cloud, _, _ = read_point_cloud(r"data\test\Cars.las")
buildings_df_test, buildings_point_cloud, _, _ = read_point_cloud(r"data\test\Buildings.las")
tree_point_cloud = add_dimension(tree_point_cloud, 'Class_Label', 'int')
for label_name in label:
    tree_point_cloud.Class_Label = all_predictions_df[all_predictions_df["label"]==1][label_name].values
    tree_point_cloud.write(f'{label_name}_Trees.las')
    ground_point_cloud.Class_Label = all_predictions_df[all_predictions_df["label"]==0][label_name].values
    ground_point_cloud.write(f'{label_name}_Ground.las')
    car_point_cloud.Class_Label = all_predictions_df[all_predictions_df["label"]==2][label_name].values
    car_point_cloud.write(f'{label_name}_Cars.las')
    buildings_point_cloud.Class_Label = all_predictions_df[all_predictions_df["label"]==3][label_name].values
    buildings_point_cloud.write(f'{label_name}_Buildings.las')


