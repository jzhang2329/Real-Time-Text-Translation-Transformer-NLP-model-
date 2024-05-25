from flask import Flask, jsonify, request
from t import use

app = Flask(__name__)



# @app.route('/api/data', methods=['GET'])
# def get_data():
#     return jsonify({"message": "Hello from Flask"})
#
# @app.route('/api/data', methods=['POST'])
# def post_data():
#     data = request.json
#     return jsonify({"received_data": data})

# @app.route('/a', methods=['GET'])
# def homepage():
#     return 'Hello World!'
# homepage()

@app.route('/result', methods=['GET'])
def service(input):
    return use(input)





if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)

