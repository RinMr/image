from flask import Flask, json, jsonify, request, render_template, send_from_directory, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import time
import base64

app = Flask(__name__)
app.secret_key = 'bouhurin'
app.jinja_env.globals.update(json=json, max=max, zip=zip)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

history_log = []
old_history_log = []
CLASS_LABELS = ['Cat: Angry', 'Cat: Sad', 'Cat: Happy', 'Dog: Angry', 'Dog: Sad','Dog: Happy', 'wolf: Angry', 'wolf: Sad', 'wolf: Happy', 'monkey: Angry', 'monkey: Sad', 'monkey: Happy', 'rion: Angry', 'rion: Sad', 'rion: Happy', ]

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    password = db.Column(db.String(120), nullable=False)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(120), nullable=False)
    labels = db.Column(db.Text, nullable=False)
    probs = db.Column(db.Text, nullable=False)
    emotions = db.Column(db.Text, nullable=True)
    message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

app.jinja_env.globals.update(zip=zip)
@app.template_filter('zip')
def zip_filter(a, b):
    return zip(a, b)

def load_models():
    animal_model = load_model('models/my_trained_model.keras')
    label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))
    emotion_model = load_model('models/emotion_model.keras')
    return animal_model, label_encoder, emotion_model

animal_model, label_encoder, emotion_model = load_models()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(image_path, user_id):
    global history_log, old_history_log

    processed_image = preprocess_image(image_path)

    animal_predictions = animal_model.predict(processed_image)[0]

    emotion_predictions = emotion_model.predict(processed_image)[0]

    all_labels = ['cat', 'dog' ,'wolf', 'monkey', 'rion']
    all_probs = (animal_predictions * 100).tolist()
    label = all_labels[np.argmax(animal_predictions)]
    confidence = all_probs[np.argmax(animal_predictions)]

    emotion_labels = label_encoder.classes_
    emotion_index = np.argmax(emotion_predictions)
    emotion_label = emotion_labels[emotion_index]
    emotion_confidence = emotion_predictions[emotion_index] * 100
    emotion_details = dict(zip(emotion_labels, (emotion_predictions * 100).tolist()))

    if 'cat' in label.lower() and confidence >= 30:
        message = "これは猫です。"
    elif 'dog' in label.lower() and confidence >= 30:
        message = "これは犬です。"
    elif 'wolf' in label.lower() and confidence >= 30:
        message = "これは狼です。"
    elif 'monkey' in label.lower() and confidence >= 30:
        message = "これは猿です。"
    elif 'rion' in label.lower() and confidence >= 30:
        message = "これはライオンです。"                        
    else:
        message = "画像が識別できませんでした。"

    new_image = Image(
        user_id=user_id,
        filename=os.path.basename(image_path),
        labels=json.dumps(all_labels),
        probs=json.dumps(all_probs),
        emotions=json.dumps(emotion_details),
        message=message,
    )
    db.session.add(new_image)
    db.session.commit()

    return label, confidence, message, emotion_details, all_labels, all_probs

@app.route('/')
def index():
    return redirect(url_for('register'))

@app.route('/home')
def home():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if not user:
        return redirect(url_for('login'))

    return render_template('home.html', user=user, history=history_log, timestamp=int(time.time()), zip=zip)

# ログイン
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="名前またはパスワードが正しくありません。")

    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image_path = os.path.join('uploads', secure_filename(file.filename))
        file.save(image_path)

        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('login'))

        animal_label, animal_confidence, message, emotion_details, animal_labels, animal_probs = predict_image(image_path, user_id)

        _, img_encoded = cv2.imencode('.png', cv2.imread(image_path))
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        predictions = list(zip(animal_labels, animal_probs))

        return render_template(
            'decision.html',
            message=message,
            image=f'<img src="data:image/png;base64,{img_base64}" alt="予測画像" />',
            emotion_details=emotion_details,
            predictions=predictions,
            top_label=animal_label
        )
    return render_template('home.html', history=history_log, timestamp=int(time.time()), zip=zip)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            error_message = "ユーザー名が既に使用されています"
            return redirect(url_for('register', error=error_message))
        
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/history')
def history():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    user_images = Image.query.filter_by(user_id=user_id).order_by(Image.timestamp.desc()).all()
    return render_template('history.html', history=user_images, timestamp=int(time.time()))

@app.route('/old_history')
def old_history():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    user_images = Image.query.filter_by(user_id=user_id).order_by(Image.timestamp.desc()).all()

    # 5件目以降の履歴を抽出
    old_history_images = user_images[5:] if len(user_images) > 5 else []

    return render_template(
        'old_history.html',
        old_history=old_history_images,
        timestamp=int(time.time())
    )

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

