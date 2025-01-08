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
    # print(f'Feautures = {features}')
    result = make_prediction(features)
    app.logger.debug(f"Result: {result}")
    return render_template('index.html', result=result)



if __name__ == '__main__':
    app.run(debug=True)

