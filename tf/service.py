from flask import Flask, jsonify, request
from t import use

app = Flask(__name__)

@app.route('/result', methods=['GET'])
def service(input):
    return use(input)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)

