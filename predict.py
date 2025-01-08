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

#dict with true output(malign tumor)
true_dict = {
    'mean_radius': 12.470,
    'mean_texture': 18.60,
    'mean_perimeter': 81.09,
    'mean_area': 481.9,
    'mean_smoothness': 0.09965,
    'mean_compactness': 0.10580,
    'mean_concavity': 0.080050,
    'mean_concave_points': 0.038210,
    'mean_symmetry': 0.1925,
    'mean_fractal_dimension': 0.06373,
    'radius_error': 0.3961,
    'texture_error': 1.0440,
    'perimeter_error': 2.4970,
    'area_error': 30.290,
    'smoothness_error': 0.006953,
    'compactness_error': 0.019110,
    'concavity_error': 0.027010,
    'concave_points_error': 0.010370,
    'symmetry_error': 0.01782,
    'fractal_dimension_error': 0.003586,
    'worst_radius': 14.97,
    'worst_texture': 24.64,
    'worst_perimeter': 96.05,
    'worst_area': 677.9,
    'worst_smoothness': 0.14260,
    'worst_compactness': 0.23780,
    'worst_concavity': 0.26710,
    'worst_concave_points': 0.10150,
    'worst_symmetry': 0.3014,
    'worst_fractal_dimension': 0.08750
}

#dict with false output(benign tumor)
false_dict = {
    'mean_radius': 0.550908,
    'mean_texture': 0.392289,
    'mean_perimeter': 0.538341,
    'mean_area': 0.411739,
    'mean_smoothness': 0.338178,
    'mean_compactness': 0.286008,
    'mean_concavity': 0.253046,
    'mean_concave_points': 0.395179,
    'mean_symmetry': 0.221570,
    'mean_fractal_dimension': 0.097936,
}

if __name__ == "__main__":
    #test the make_prediction function with sample data
    make_prediction(false_dict)
