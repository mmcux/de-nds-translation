from flask import Flask, request, jsonify, render_template
from translate_input import translate_to_platt
import logging
logging.basicConfig(filename="error.log",level=logging.DEBUG)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['POST'])

def predict():
    input_sentence = request.form['hochdeutsch']

    output_sentence = translate_to_platt(input_sentence)

    app.logger.info('Translation \n %s \n %s', input_sentence, output_sentence)
    return render_template('index.html', input_text = input_sentence, prediction_text = output_sentence)

if __name__ == "__main__":
    app.run(debug=True)
