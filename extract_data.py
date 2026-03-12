from point_cloud import *
from sklearn.model_selection import train_test_split
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