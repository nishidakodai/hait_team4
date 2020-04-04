import os
import numpy as np
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import send_from_directory
from tensorflow.keras.models import load_model
import cv2

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pred, url, rate = predict(filename)
            return render_template('predict.html', predict=pred, url=url, imgpath=url_for('static', filename=filename), rate=rate)
    return render_template('top.html')

@app.route('/uploads/<filename>')
# 予測結果を表示する
def predict(filename):
    model = load_model('./../model/vgg16_food_9class_1.h5')
    resized_img = resizeImg(filename)
    predict_rate = model.predict(resized_img)
    predicted_index = np.argmax(predict_rate)
    rate = predict_rate[0][predicted_index]
    if predicted_index == 0:
        return "カリフォルニアロール", "https://cookpad.com/search/%E3%82%AB%E3%83%AA%E3%83%95%E3%82%A9%E3%83%AB%E3%83%8B%E3%82%A2%E3%83%AD%E3%83%BC%E3%83%AB", rate
    elif predicted_index == 1:
        return "チャーハン", "https://cookpad.com/search/%E3%83%81%E3%83%A3%E3%83%BC%E3%83%8F%E3%83%B3", rate
    elif predicted_index == 2:
        return "カレーライス", "https://cookpad.com/search/%E3%82%AB%E3%83%AC%E3%83%BC%E3%83%A9%E3%82%A4%E3%82%B9", rate
    elif predicted_index == 3:
        return "牛丼", "https://cookpad.com/search/%E7%89%9B%E4%B8%BC", rate
    elif predicted_index == 4:
        return "唐揚げ", "https://cookpad.com/search/%E5%94%90%E6%8F%9A%E3%81%92", rate
    elif predicted_index == 5:
        return "肉じゃが", "https://cookpad.com/search/%E8%82%89%E3%81%98%E3%82%83%E3%81%8C", rate
    elif predicted_index == 6:
        return "パスタ", "https://cookpad.com/search/%E3%83%91%E3%82%B9%E3%82%BF", rate
    elif predicted_index == 7:
        return "ご飯", "https://www.google.com/search?q=%E3%81%94%E9%A3%AF+%E7%82%8A%E3%81%8D%E6%96%B9&oq=%E3%81%94%E9%A3%AF%E3%80%80%E7%82%8A%E3%81%8D%E6%96%B9&aqs=chrome..69i57j0l7.3416j0j7&sourceid=chrome&ie=UTF-8", rate
    elif predicted_index == 8:
        return "ローストビーフ", "https://cookpad.com/search/%E3%83%AD%E3%83%BC%E3%82%B9%E3%83%88%E3%83%93%E3%83%BC%E3%83%95", rate

def resizeImg(filename):
    input_img = cv2.imread(UPLOAD_FOLDER + filename)
    resized_img = cv2.resize(input_img , dsize=(224, 224))

    resized_img = resized_img[None, ...]

    resized_img=resized_img.astype('float32')/255.0
    resized_img=resized_img.reshape((1,224,224,3))

    cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    return resized_img

if __name__ == "__main__":
    app.run(port=8000, debug=True)