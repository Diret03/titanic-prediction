from flask import Flask, render_template, request
from predict import make_prediction

app = Flask(__name__, template_folder='templates')


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    result = make_prediction(features)
    app.logger.debug(f"Result: {result}")
    return render_template('index.html', result=result, form_data=features)



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

