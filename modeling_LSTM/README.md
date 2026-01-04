# Ini Proses Implementasi Algoritma LSTMs varian model terbaik, yaitu Bi-LSTM 1 Layer + FastText
Link FastText yang digunakan:
https://drive.google.com/file/d/12o2eoFprLsTo-EvKETNNBlaNKVhoqNMh/view?usp=drive_link

# Rangkuman Parameter LSTM (Bi-LSTM 1 Layer + FastText):
Data Input:
- Pembagian Data = 70% Train : 20% Val : 10% Test
- Panjang Sekuens (Max Len)	= 15 token (kata)
- Word Embedding	= Pre-trained FastText (300 dimensi)
Arsitektur Model:
- Embedding Layer Trainable = True
- Hidden Layer = Bidirectional LSTM (64 unit)
- Dense Layer =	64 unit (Aktivasi ReLU, L2 Regularizer)
- Output Layer	= 10 unit (Aktivasi Softmax)
Regularisasi:
- Dropout	= 0.2 (pada lapisan Bi-LSTM dan setelah Batch Norm)
- Batch Normalization	= Ya (setelah layer Bi-LSTM)
Konfigurasi Pelatihan:
- Optimizer	RMSprop (Learning Rate = 0.0001)
- Fungsi Loss	Sparse Categorical Crossentropy
- Epoch	15
- Batch Size	30





