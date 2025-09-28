""" 
>. Final Task - Category prediction based on 'products.csv' data
>. Author: Danilo Jelovac


>>. Goal:
-------------
- Our goal here is to propperly train the chosen model, `Naive Bayes`
to correctly predict and sort category labels based on the product_label
we're feeding it.

>>. Requirements:
------------
- pandas
- skicit-learn

"""

# -------------------
# Importing libraries:
# -------------------

import pandas as pd    # --loading data...
# --
from sklearn.feature_extraction.text import TfidfVectorizer    # --neccessary tools...
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# --
from sklearn.naive_bayes import MultinomialNB    # --our chosen model...
# --
import joblib    # --for saving the model


# ---------------------------------
FOLDER_NAME = "ml_data"
FILE_NAME = "products_cleaned.csv"
# ---------------------------------


# -------------------
# Loading the dataset:
# -------------------

try:
    data = pd.read_csv(f"{FOLDER_NAME}/{FILE_NAME}")
    print(f">. '{FILE_NAME}' loaded successfuly!")
except Exception:
    print("Something went wrong... Please, check your directory and/or path.")


# --------------
# Model training:
# --------------

x_input = data['product_title']
y_output = data['category_label']

# --Separating the samples:

X_train, X_test, y_train, y_test = train_test_split(
    x_input, y_output, test_size=0.2, random_state=42,
    stratify=y_output)

# --Pipeline:

pipeline = Pipeline([
    ("Tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
    ("Naive Bayes", MultinomialNB())
])
pipeline.fit(X_train, y_train)

# --Prediction and classification report:

y_pred = pipeline.predict(X_test)

print("\n>>. MODEL -> ['Naive Bayes'] classification report:\n", "-"*50)
print(classification_report(y_test, y_pred), "\n")

# --Saving the model:

# ------------------------------------
MODEL_FOLDER_NAME = 'model'
MODEL_NAME = 'category_label_model.pkl'
PATH = f'{MODEL_FOLDER_NAME}/{MODEL_NAME}'
# ------------------------------------

try:
    joblib.dump(pipeline, PATH)
    print(f"\n>>>. Model '{MODEL_NAME}' successfuly created!")
    print(f">>>. Model saved in ./root/{PATH}.")
except Exception:
    print("Something went wrong... Please, check your code and/or path.")