# Import Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers

# Inisialiasi
texts = df['judul_bersih'].values

# --- 1. Label Encoder ---
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['kategori_utama'])

# --- 2. Tokenizer dan Padding
# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# Panjang maksimal
max_len = 15
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# --- 3. Split Data ---
# Split pertama: 70% train, 30% sisanya
X_train, X_temp, y_train, y_temp = train_test_split(
    X, labels, test_size=0.3, random_state=42, stratify=labels
)
# Split kedua: 20% val dan 10% test dari sisa 30%
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp
)

# --- 4. Word Embedding
embedding_path = 'cc.id.300.vec' # File bisa didownload pada README.md
embedding_index = {}
with open(embedding_path, encoding='utf-8', errors='ignore') as f:
    # Lewati baris pertama (header) karena isinya cuma info dimensi (bukan vektor)
    next(f)

    for line in f:
        values = line.split()
        word = values[0]
        # Pastikan mengambil nilai vektornya saja
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        except ValueError:
            # Lewati jika ada baris yang error/korup (jarang terjadi tapi buat jaga-jaga)
            continue

print(f'Berhasil memuat {len(embedding_index)} vektor kata dari FastText.')

# Buat matrix embedding berdasarkan vocab tokenizer
embedding_dim = 300
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

# Hitung berapa kata yang ketemu (Match) vs tidak ketemu (Miss)
hits = 0
misses = 0

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # Kata ditemukan di FastText
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        # Kata tidak ditemukan (OOV - Out of Vocabulary)
        misses += 1

print(f"Kata yang ditemukan (Hits): {hits}")
print(f"Kata yang tidak ditemukan (Misses): {misses}")

# --- 5. Bangun Arsitektur Model ---
model = Sequential()
model.add(Input(shape=(max_len,)))  # Tambahkan baris ini
model.add(Embedding(
    input_dim=len(word_index) + 1,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=True
))
model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64, recurrent_dropout=0.2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# --- 6. Pelatihan Model --- 
# Inisialisasi optimizer dengan learning rate tertentu
optimizer = RMSprop(learning_rate=0.0001)
# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)
# Latih model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=30
)

# --- 7. Evaluasi Model ---
# Membuat Grafik Akurasi dan Loss
# Plot Akurasi
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Prediksi
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
# Classification report
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - bi-LSTM with Glove')
plt.show()

# --- 8. Menyimpan Model ---
# Simpan Model LSTM
model.save("model_lstm.h5")
# Simpan Tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
# Simpan LabelEncoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
# Simpan max_len (untuk padding saat prediksi nanti)
with open("max_len.pkl", "wb") as f:
    pickle.dump(max_len, f)
