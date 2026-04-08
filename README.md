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

## 📸 Screenshot & Demo
Naive Bayes
- Confusion Matrix
  
  <img width="526" height="471" alt="image" src="https://github.com/user-attachments/assets/5a2a4977-caa6-4ac4-bd75-7e6c2dc34006" />
- Hasil Pengujian
  
  <img width="543" height="183" alt="image" src="https://github.com/user-attachments/assets/d8ff55b2-bcfa-45e1-8579-95527077009e" />
LSTM
- Grafik Akurasi dan Loss
  
  <img width="537" height="285" alt="image" src="https://github.com/user-attachments/assets/2e42650b-0512-4734-8952-42f5ecce1fee" />
  
  <img width="528" height="291" alt="image" src="https://github.com/user-attachments/assets/3797fbaa-b545-40e0-bfc0-394dd246eb96" />
- Confusion Matrix
  
  <img width="523" height="452" alt="image" src="https://github.com/user-attachments/assets/08a1be3e-2278-48b5-a4e4-5fee63ec6518" />
- Hasil Pengujian
  
  <img width="535" height="603" alt="image" src="https://github.com/user-attachments/assets/4f44a409-67db-44c8-89d0-818fb489b0ac" />




