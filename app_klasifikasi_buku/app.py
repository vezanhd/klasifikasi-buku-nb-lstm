from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Inisialisasi Flask
app = Flask(__name__)

# === Load tools ===
model = load_model('model/model_lstm.h5')
tokenizer = pickle.load(open('model/tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('model/label_encoder.pkl', 'rb'))
max_len = pickle.load(open('model/max_len.pkl', 'rb'))

# === Mapping kode kategori ke nama deskriptif ===
kategori_mapping = {
    '000': "000 - Komputer, informasi dan referensi umum",
    '100': "100 - Filsafat dan Psikologi",
    '200': "200 - Agama",
    '300': "300 - Ilmu Sosial",
    '400': "400 - Bahasa",
    '500': "500 - Sains dan matematika",
    '600': "600 - Teknologi",
    '700': "700 - Kesenian dan rekreasi",
    '800': "800 - Sastra",
    '900': "900 - Sejarah dan Geografi"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    hasil = None
    judul = ""  # ← Tambahkan ini sebagai nilai default
    if request.method == 'POST':
        judul = request.form['judul']

        # Tokenisasi dan padding
        sequences = tokenizer.texts_to_sequences([judul])
        padded = pad_sequences(sequences, maxlen=max_len, padding='post')

        # Prediksi
        pred = model.predict(padded)
        label_pred = np.argmax(pred, axis=1)
        kode_kategori = label_encoder.inverse_transform(label_pred)[0]
        nama_kategori = kategori_mapping.get(kode_kategori, "Kategori tidak diketahui")
        hasil = f"{nama_kategori}"

    return render_template('index.html', hasil=hasil, judul=judul)

if __name__ == '__main__':
    app.run(debug=True)