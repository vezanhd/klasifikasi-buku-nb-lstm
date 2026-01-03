Ini Proses Implementasi Algoritma Naive Bayes varian model terbaik, yaitu Deep Feature Weighting Naïve Bayes (DFWNB)

# Rangkuman Parameter Naive Bayes (DFWNB):
Penanganan Data = Undersampling (Data Seimbang)	==> Target ~500 data per kelas

Pembagian Data (Split) = 80% Training : 20% Testing	==> Menggunakan Stratified Sampling

Metode Ekstraksi Fitur	= TF-IDF (Term Frequency-Inverse Document Frequency	==> Mengubah teks menjadi vektor bobot

Seleksi Fitur	= Chi-Square (X^2)	==> Memilih k=100 fitur terbaik

Pembobotan Fitur (Deep Weighting)	= W_terpilih=2, W_lainnya=1	==> Fitur hasil seleksi diberi bobot ganda.


