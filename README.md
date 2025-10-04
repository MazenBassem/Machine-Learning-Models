# Machine Learning Models

> Practical, ready-to-run Jupyter notebooks and datasets for hands-on machine learning and data science.

---

![Stars](https://img.shields.io/github/stars/MazenBassem/Machine-Learning-Models?style=social)
![Last Commit](https://img.shields.io/github/last-commit/MazenBassem/Machine-Learning-Models)

---

## Overview

Explore real-world machine learning problems with concise code, clear explanations, and visualizations. Each task demonstrates end-to-end workflows—from data prep to model evaluation—for educational and practical learning.

---

## 🗂️ Project Structure

```
Machine-Learning-Models/
│
├── Task 1: Student Score Prediction/
│   ├── Task_1_Student_Score_Prediction.ipynb
│   ├── StudentPerformanceFactors.csv
│   └── utils.py
│
├── Task 2: Customer Segmentation/
│   ├── Task_2_Customer_Segmentation.ipynb
│   ├── Mall_Customers.csv
│   └── utils.py
│
├── Task 3: Forest Cover Type Classification/
│   └── Task_3_Forest_Cover_Type_Classification.ipynb
│
└── README.md
```

---

## 📂 Tasks & Models

### 📝 1. Student Score Prediction
- **Notebook:** [View](Task%201:%20Student%20Score%20Prediction/Task_1_Student_Score_Prediction.ipynb) | **Dataset:** [CSV](Task%201:%20Student%20Score%20Prediction/StudentPerformanceFactors.csv)
- **Goal:** Predict student exam scores from various performance factors.
- **Key Models:** Linear Regression, Ridge Regression, Polynomial Regression
- **Highlights:** Data cleaning, feature engineering, regression analysis, metric comparison  

<table>
  <tr>
    <td><img src="images/Score%20vs%20hours%20Studied.jpg" width="120"></td>
    <td><img src="images/Polynomial%20Degree%20vs%20MSE.jpg" width="120"></td>
    <td><img src="images/lambda%20vs%20MSE.png" width="120"></td>
  </tr>
</table>

---

### 👥 2. Customer Segmentation
- **Dataset:** [CSV](Task%202:%20Customer%20Segmentation/Mall_Customers.csv) | **Utils:** [Python](Task%202:%20Customer%20Segmentation/utils.py)
- **Goal:** Cluster mall customers for marketing insights.
- **Key Models:** K-Means, DBSCAN
- **Highlights:** Data preprocessing, cluster visualization, spending analysis, unsupervised learning  

<table>
  <tr>
    <td><img src="images/kmeans%20clusters%20with%20centroids.jpg" width="120"></td>
    <td><img src="images/DBSCAN%20Clustering.jpg" width="120"></td>
    <td><img src="images/clusters%20averages%20(dbscan%20vs%20kmeans).jpg" width="120"></td>
    <td><img src="images/Elbow%20methof%20for%20DBSCAN.jpg" width="120"></td>
  </tr>
</table>

---

### 🌲 3. Forest Cover Type Classification
- **Notebook:** [View](Task%203:%20Forest%20Cover%20Type%20Classification/Task_3_Forest_Cover_Type_Classification.ipynb)
- **Goal:** Predict forest cover type using environmental features.
- **Key Models:** XGBoost, Random Forest, Decision Tree
- **Highlights:** Data cleaning, feature analysis, model comparison, hyperparameter tuning  

<table>
  <tr>
    <td><img src="images/random%20forest%20perfomance.jpg" width="140"></td>
    <td><img src="images/xgboost%20perfomance.jpg" width="140"></td>
    <td><img src="images/Compare%20Model%20Perfomance.jpg" width="120"></td>
  </tr>
</table>

---

## 🚀 Quickstart

> **Tip:** You can instantly try the notebooks online using [Google Colab](https://colab.research.google.com/)—no setup needed!

```bash
git clone https://github.com/MazenBassem/Machine-Learning-Models.git
cd Machine-Learning-Models
jupyter notebook
```
- Use Jupyter Notebook/Lab (Python 3.10+).  
- Install dependencies as prompted in each notebook.

---


## 🤝 Contribute

Ideas, notebooks, or improvements?  
Open an issue or pull request—collaboration is welcome!

---

> ⭐ Enjoy the repo? Give it a star and share!
