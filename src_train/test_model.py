""" 
>. Final Task - Category prediction based on 'products.csv' data
>. Author: Danilo Jelovac


>>. Goal:
-------------
- After successfuly creating a model, this script will be for testing
it's ability in sorting 'category_label' when given a product name.

>>. Requirements:
------------
- skicit-learn

"""

# -------------------
# Importing libraries:
# -------------------

import joblib


# ------------------------------------------
MODEL_FOLDER_NAME = 'model'
MODEL_FILE_NAME = 'category_label_model.pkl'
# ------------------------------------------


# -----------------
# Getting our model:
# -----------------

pipeline = joblib.load(f"{MODEL_FOLDER_NAME}/{MODEL_FILE_NAME}")

# ----------
# Model test:
# ----------

print("""
======= MODEL TEST SCRIPT =======
>. Enter the product title to test model's prediction.
>. Enter 'exit' at any time to leave the testing app.
      """)

while True:
    
    user_input = input(">>. Enter product title: ")
    
    if user_input.lower() == 'exit':
        print("\n>>. Thx, goodbye!\n")
        break
    
    prediction = pipeline.predict([user_input])[0]
    print(f"Predicted category label -> '{prediction}'.\n", "-"*50)
