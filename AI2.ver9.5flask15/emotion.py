import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def load_emotion_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    # サポートする画像拡張子
    supported_extensions = {'.jpg', '.jpeg', '.png', '.jfif'}
    
    for category in ['Cat', 'Dog', 'Wolf', 'Monkey', 'Rion']:  # 'Cat' と 'Dog' のカテゴリを追加
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for emotion in ['angry', 'sad', 'Happy']:
                emotion_dir = os.path.join(category_dir, emotion)
                if os.path.isdir(emotion_dir):
                    for img_file in os.listdir(emotion_dir):
                        # ファイルの拡張子をチェック
                        ext = os.path.splitext(img_file)[1].lower()
                        if ext in supported_extensions:
                            img_path = os.path.join(emotion_dir, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.resize(img, img_size) / 255.0  # 正規化
                                images.append(img)
                                labels.append(f"{category}_{emotion}")  # 例: 'Cat_angry'
    return np.array(images, dtype=np.float32), np.array(labels)

# データのロード
data_dir = 'C:\\Users\\user\\Desktop\\GitHub\\allimages\\emotion'
img_size = (128, 128)
X, y = load_emotion_data(data_dir, img_size)

# ラベルのエンコーディング
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# **感情識別モデルの構築**
def create_emotion_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # クラス数に応じた出力
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# モデル作成
input_shape = (img_size[0], img_size[1], 3)
num_classes = len(label_encoder.classes_)
emotion_model = create_emotion_model(input_shape, num_classes)

# **トレーニング**
batch_size = 32
epochs = 20
emotion_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

# モデルとエンコーダの保存
model_dir = 'C:\\Users\\user\\Desktop\\GitHub\\AI2.ver9.5flask15\\models'
os.makedirs(model_dir, exist_ok=True)
emotion_model.save(os.path.join(model_dir, 'emotion_model.keras'))
with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print("感情識別モデルとラベルエンコーダが保存されました。")
