# Klasifikasi Kategori Buku Berdasarkan Judul Menggunakan Naïve Bayes & LSTM

**Lampiran Skripsi** • Teknik Informatika • Universitas Lampung  
**Akurasi terbaik: 79.30%** (LSTM)

Proyek ini mengembangkan dua model klasifikasi teks untuk mengkategorikan buku berdasarkan judulnya (studi kasus: UPA Perpustakaan Universitas Lampung).

## 🎯 Fitur Utama
- Preprocessing teks (cleaning, tokenisasi, TF-IDF)
- Model **Naïve Bayes** (scikit-learn)
- Model **LSTM** (TensorFlow/Keras)
- Perbandingan performa kedua model
- Web app sederhana dengan **Flask** (deploy model terbaik)

## 📊 Hasil
| Model          | Akurasi   | Keterangan                  |
|----------------|-----------|-----------------------------|
| Naïve Bayes    | ~69%      | Cepat & ringan              |
| **LSTM**       | **79.30%**| Lebih baik pada data kompleks |

## 🛠️ Tech Stack
- **Python** • Google Colab
- scikit-learn • TensorFlow/Keras • pandas • numpy • matplotlib
- Flask (Deployment)

## 📁 Struktur Folder
dataset/                  ← Data yang dipakai
datapreparation/          ← Script cleaning & preprocessing
modeling_nb/              ← Model Naïve Bayes
modeling_LSTM/            ← Model LSTM + training
app_klasifikasi_buku/     ← Codingan Web app Flask
