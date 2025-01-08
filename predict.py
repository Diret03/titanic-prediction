import tensorflow as tf
import numpy as np
import joblib

# Get prefitted scaler
# This scaler was fitted previously based on the train data from sklearn breast cancer dataset
scaler = joblib.load('./models/titanic-scaler.joblib')


def load_model():
    #Load sequential tensorflow model created and trained previously
    loaded_model = tf.keras.models.load_model('./models/RNA_titanic_model.h5')
    print("Model loaded.")
    return loaded_model


model = load_model()


def transform_data(features_dict):
    values = features_dict.values()
    arr = np.array(list(values), dtype=float)
    # print(f"Array: {arr}")

    # reshape array to 2D: one sample with 7 parameters
    arr = arr.reshape(1, -1)

    #normalize data [0,1]
    arr = scaler.transform(arr)
    print(f"Array resized: {arr}")
    return arr


def make_prediction(features):
    test_data = transform_data(features)
    print(f"Test_Data: {test_data}")

    prediction = model.predict(test_data)
    # binary labeling
    prediction = (prediction > 0.5)

    print(f"Prediction: {prediction[0][0]}")

    return prediction[0][0]

# dict with true output (survived passenger)
true_dict = {
    'pclass': 1,              # First class
    'sex': 0,                 # Female
    'age': 5,                 # Age group 26-40
    'sibsp': 0,               # No siblings or spouses aboard
    'parch': 1,               # 1 parent or child aboard
    'fare': 71.2833,          # High fare
    'embarked': 2             # Embarked from Cherbourg
}

# dict with false output (dead passenger)
false_dict = {
    'pclass': 3,              # Third class
    'sex': 1,                 # Male
    'age': 4,                 # Age group 19-25
    'sibsp': 1,               # 1 sibling or spouse aboard
    'parch': 0,               # 0 parents or children aboard
    'fare': 7.25,             # Low fare
    'embarked': 1             # Embarked from Southampton
}

if __name__ == "__main__":
    #test the make_prediction function with sample data
    make_prediction(false_dict)
