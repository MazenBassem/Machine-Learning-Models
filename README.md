# Machine-Learning-Models
A comprehensive collection of machine learning models and Jupyter notebooks for educational and research purposes.

## 📚 Repository Structure

This repository is organized into the following categories:

```
notebooks/
├── supervised-learning/     # Classification and regression models
├── unsupervised-learning/   # Clustering, PCA, and anomaly detection
├── deep-learning/           # Neural networks and deep learning architectures
├── nlp/                     # Natural Language Processing models
├── computer-vision/         # Image processing and computer vision
├── reinforcement-learning/  # RL algorithms and agents
├── data-preprocessing/      # Data cleaning and feature engineering
└── misc/                    # Other ML-related notebooks
```

## 📤 How to Upload Your Notebooks

### Option 1: Using GitHub Web Interface

1. Navigate to the appropriate category folder (e.g., `notebooks/supervised-learning/`)
2. Click on "Add file" → "Upload files"
3. Drag and drop your `.ipynb` file(s) or click to browse
4. Add a commit message describing your notebook
5. Click "Commit changes"

### Option 2: Using Git Command Line

1. Clone the repository:
   ```bash
   git clone https://github.com/MazenBassem/Machine-Learning-Models.git
   cd Machine-Learning-Models
   ```

2. Create a new branch for your notebook:
   ```bash
   git checkout -b add-your-notebook-name
   ```

3. Copy your notebook to the appropriate category folder:
   ```bash
   cp /path/to/your/notebook.ipynb notebooks/appropriate-category/
   ```

4. Add and commit your changes:
   ```bash
   git add notebooks/appropriate-category/your-notebook.ipynb
   git commit -m "Add: [Brief description of your notebook]"
   ```

5. Push your branch and create a pull request:
   ```bash
   git push origin add-your-notebook-name
   ```

### Option 3: Using Google Drive/Colab

If your notebook is on Google Drive or Colab:

1. Download the notebook as `.ipynb` file (File → Download → Download .ipynb)
2. Follow Option 1 or Option 2 above to upload it

## 📝 Notebook Guidelines

### Naming Convention
Use descriptive, lowercase names with hyphens:
- ✅ Good: `linear-regression-housing-prices.ipynb`
- ✅ Good: `cnn-mnist-classification.ipynb`
- ❌ Bad: `Untitled1.ipynb`
- ❌ Bad: `MyNotebook.ipynb`

### Content Guidelines
Your notebook should include:
- Clear title and description
- Import statements and dependencies
- Data loading and exploration
- Model implementation and training
- Results and visualization
- Conclusions and next steps
- Comments explaining key steps

### Before Uploading
- Clear all output cells if they contain large outputs (optional)
- Ensure the notebook runs from top to bottom without errors
- Remove any personal or sensitive information
- Document any required datasets or dependencies

## 🎯 Categories

### Supervised Learning
Classification and regression algorithms including linear models, tree-based methods, and ensemble techniques.

### Unsupervised Learning
Clustering, dimensionality reduction, and pattern discovery algorithms.

### Deep Learning
Neural networks, CNNs, RNNs, LSTMs, Transformers, and other deep learning architectures.

### Natural Language Processing
Text processing, sentiment analysis, language models, and NLP applications.

### Computer Vision
Image classification, object detection, segmentation, and visual recognition tasks.

### Reinforcement Learning
RL algorithms, agents, and interactive learning systems.

### Data Preprocessing
Data cleaning, feature engineering, EDA, and data preparation techniques.

### Miscellaneous
Time series, recommender systems, optimization, and other ML topics.

## 🤝 Contributing

We welcome contributions! Whether you have a simple tutorial or an advanced implementation, feel free to share your work. This repository is meant to be a collaborative learning resource for the ML community.

## 📄 License

This repository is for educational purposes. Please ensure you have the right to share any code or data you upload, and respect any licensing requirements of the datasets or libraries you use.
