from flask import Flask, jsonify, render_template, request, redirect, url_for
import os
from BackgroundSwitcher import BackgroundSwitcher

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/images'

app.backgroundSwitcher = BackgroundSwitcher(
    images_path=app.config['UPLOAD_FOLDER'],
    device="cuda",
    checkpoints_path="./checkpoints",
    use_sd_inpainting=False,
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/swap', methods=['POST'])
def swap_backgrounds():
    file1 = request.files['file1']
    file2 = request.files['file2']
    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
    file1.save(file1_path)
    file2.save(file2_path)

    # Perform background swapping logic
    url1, url2 = app.backgroundSwitcher.switch_foregrounds(file1.filename, file2.filename)
    return jsonify({'result1': url1, 'result2': url2})

if __name__ == '__main__':
    app.run()