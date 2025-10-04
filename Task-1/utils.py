# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Plot training and CV MSEs vs polynomial degree
def plot_train_cv_mses(degrees, train_mses, cv_mses, title):
    plt.figure(figsize=(8, 6))
    plt.plot(degrees, train_mses, marker='o', label='Train MSE')
    plt.plot(degrees, cv_mses, marker='o', label='CV MSE')
    plt.title(title)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.xticks(degrees)
    plt.legend()
    plt.grid(True)
    plt.show()

# Clean full dataset
def clean(data):
    y = data["Exam_Score"]
    X = data.drop("Exam_Score", axis=1, errors='ignore')

    imputation_map = {
        'Parental_Education_Level': ['Previous_Scores', 'Attendance', 'Parental_Involvement', 'Access_to_Resources'],
        'Teacher_Quality': ['Previous_Scores', 'Access_to_Resources'],
        'Distance_from_Home': ['Parental_Involvement', 'Access_to_Resources', 'Attendance']
    }

    for target_col, group_cols in imputation_map.items():
        if target_col in X.columns:
            existing_group_cols = [col for col in group_cols if col in X.columns]
            if existing_group_cols:
                overall_mode = X[target_col].mode()[0] if not X[target_col].mode().empty else 'Unknown'
                X[target_col] = X.groupby(existing_group_cols)[target_col].transform(
                    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else overall_mode)
                )

    return X, y

# Clean dataset to keep only 'Hours_Studied'
def clean_hours_only(data):
    y = data["Exam_Score"]
    data = data.drop("Exam_Score", axis=1, errors='ignore')

    dropped = [
        'Attendance', 'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Sleep_Hours', 'Previous_Scores', 'Motivation_Level', 'Internet_Access', 'Tutoring_Sessions',
        'Family_Income', 'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Physical_Activity',
        'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    data = data.drop(dropped, axis=1, errors='ignore')
    return data, y

# Train and plot regularization parameter vs MSE
def train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree=1, baseline=None):
    train_mses, cv_mses, models = [], [], []

    for reg_param in reg_params:
        model = Ridge(alpha=reg_param)
        model.fit(x_train, y_train)
        models.append(model)

        train_mse = mean_squared_error(y_train, model.predict(x_train)) / 2
        cv_mse = mean_squared_error(y_cv, model.predict(x_cv)) / 2

        train_mses.append(train_mse)
        cv_mses.append(cv_mse)

    reg_params_str = [str(x) for x in reg_params]
    plt.plot(reg_params_str, train_mses, marker='o', c='r', label='Training MSE')
    plt.plot(reg_params_str, cv_mses, marker='o', c='b', label='CV MSE')
    if baseline is not None:
        plt.plot(reg_params_str, [baseline] * len(reg_params), linestyle='--', label='Baseline')
    plt.title("Lambda vs. Train and CV MSEs")
    plt.xlabel("Lambda")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

# Plot linear fit for Hours Studied vs Exam Score
def plot_hours_studied_vs_exam_score_linear_fit(X_train, y_train):
    hours_studied = X_train['Hours_Studied'].values.reshape(-1, 1)

    scaler = StandardScaler()
    hours_studied_scaled = scaler.fit_transform(hours_studied)

    model = LinearRegression()
    model.fit(hours_studied_scaled, y_train)

    x_plot = np.linspace(hours_studied.min(), hours_studied.max(), 100).reshape(-1, 1)
    x_plot_scaled = scaler.transform(x_plot)
    y_plot = model.predict(x_plot_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(hours_studied, y_train, label='Training Data', alpha=0.6)
    plt.plot(x_plot, y_plot, color='red', label='Linear Regression Line')
    plt.title("Linear Regression: Hours Studied vs. Exam Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot KMeans clusters
def plot_clusters(train_data_clean, predictions, centroids, i=0):
    plt.figure(figsize=(8, 6))
    plt.scatter(train_data_clean.iloc[:, 0], train_data_clean.iloc[:, 1], c=predictions, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label='Centroids')
    title = f'K-Means Clustering of Customer Data Model {i}' if i else 'K-Means Clustering of Customer Data'
    plt.title(title)
    plt.xlabel('Annual Income (k$) (Scaled)')
    plt.ylabel('Spending Score (1-100) (Scaled)')
    plt.legend()
    plt.show()

# Initialize KMeans centroids
def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    return X.iloc[randidx[:K]]

def plot_clusters(train_data_clean, predictions, centroids, i = 0):
  # Visualize the clusters and centroids
  plt.figure(figsize=(8, 6))
  plt.scatter(train_data_clean.iloc[:, 0], train_data_clean.iloc[:, 1], c=predictions, cmap='viridis', marker='o')
  plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x', label='Centroids')

  plt.title('K-Means Clustering of Customer Data')
  if i != 0:
    plt.title('K-Means Clustering of Customer Data Model ' + str(i))
  plt.xlabel('Annual Income (k$) (Scaled)')
  plt.ylabel('Spending Score (1-100) (Scaled)')
  plt.legend()
  # plt.grid(True)
  plt.show()

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids using .iloc
    centroids = X.iloc[randidx[:K]]

    return centroids

def plot_cluster_averages(cluster_summary, method_name):
    """
    Plots average feature values per cluster for a given clustering method.

    Args:
        cluster_summary (pd.DataFrame): DataFrame with 'Cluster' and average feature columns.
        method_name (str): Name of the clustering method (e.g., 'KMeans', 'DBSCAN').
    """
    # Sort clusters by average Spending Score
    sorted_clusters = cluster_summary.sort_values(by='Average Spending Score')['Cluster']
    
    # Melt the DataFrame for plotting
    melted = cluster_summary.melt(id_vars='Cluster', var_name='Metric', value_name='Average')
    
    # Ensure Cluster is treated as categorical with sorted order
    melted['Cluster'] = pd.Categorical(melted['Cluster'], categories=sorted_clusters, ordered=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='Cluster', y='Average', hue='Metric')
    plt.title(f"{method_name} Cluster Averages")
    plt.xlabel("Cluster")
    plt.ylabel("Average Value")
    plt.legend(title="Metric")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
