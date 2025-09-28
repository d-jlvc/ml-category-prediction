## ML Category Prediction

This project demonstrates how to build a machine learning model that classifies products into categories based on their titles.  


### 📂 - Project Structure

ml-category-prediction/
|
│── ml_data/
│ ├── products.csv    # Raw dataset
│ ├── products_cleaned.csv    # Cleaned dataset
│
│── model/
│ └── category_label_model.pkl    # Trained model (ignored in .gitignore)
│
│── notebooks/
│ ├── products_data_analysis.ipynb    # Data cleaning & visualization
│ ├── ml_model_eval.ipynb    # Model testing & evaluation
│
│── src_train/
│ ├── train_model.py    # Script for training & saving the model
│ ├── test_model.py    # Script for loading & testing the model
│
|
│── .venv/    # Virtual environment (ignored in .gitignore)
│── requirements.txt
│── README.md


### 🚀 - Project Workflow

1. **Data Cleaning & Analysis**
   - Used `pandas` for preprocessing
   - Visualized distributions using `matplotlib` and `seaborn`
   - Saved cleaned dataset as `products_cleaned.csv`

2. **Model Evaluation**
   - Tested multiple ML models in Jupyter Notebook:
     - Logistic Regression
     - Naive Bayes
     - Decision Tree
     - Random Forest
     - Support Vector Machine
   - Selected the best-performing model

3. **Training & Saving the Model**
   - Trained final model using `train_model.py`
   - Saved trained model as `category_label_model.pkl`

4. **Testing the Model**
   - Verified predictions with `test_model.py`


### 📦 - Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/your-username/ml-category-prediction.git
cd ml-category-prediction
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Included!

```


### ▶️ - Usage

Run Jupyter notebooks for data exploration and model evaluation:

jupyter notebooks are in "notebooks" folder, preprocessing and eval.

To train a new model: python src_train/train_model.py
To test the trained model: python src_train/test_model.py


### 📚 - Requirements

pandas
scikit-learn
matplotlib
seaborn
jupyter
ipykernel

Install them via: pip install -r requirements.txt


### 🛑 - Notes

The trained model (.pkl) is not included in the repository (see .gitignore).
To reproduce results, retrain the model using train_model.py.
