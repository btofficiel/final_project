from flask import Flask, jsonify, request, render_template
import filesave
import subprocess

app = Flask(__name__)

def build_response(class_id):
    if class_id.strip() == '0':
        return "benign"
    elif class_id.strip() == '1':
        return "malignant"
    else:
        return "error"

@app.route("/", defaults={'path': ''})
def index(path):
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            filesave.save_file(file)
            cmd = "python main.py"
            result = subprocess.check_output(cmd, shell=True)
            return jsonify({'class': build_response(result.decode())})

if __name__ == '__main__':
    app.run()
