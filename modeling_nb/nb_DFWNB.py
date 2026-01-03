# Import Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# --- 1. Seimbangkan Data ---
# Target jumlah per kategori (kecuali 700)
target_per_class = 500
# Buat list DataFrame per kategori
balanced_dfs = []
# Iterasi setiap kategori unik
for label in df['kategori_utama'].unique():
    df_kategori = df[df['kategori_utama'] == label]
    if label == '700':
        # Untuk kategori 700, biarkan jumlah asli
        balanced_dfs.append(df_kategori)
    else:
        if len(df_kategori) >= target_per_class:
            sampled_df = df_kategori.sample(n=target_per_class, random_state=42)
        else:
            sampled_df = df_kategori.sample(n=target_per_class, replace=True, random_state=42)
        balanced_dfs.append(sampled_df)
# Gabungkan semua hasil
df_seimbang = pd.concat(balanced_dfs)
# Reset index (opsional)
df_seimbang = df_seimbang.sample(frac=1, random_state=42).reset_index(drop=True)
# Cek hasil
print(df_seimbang['kategori_utama'].value_counts())

# --- 2. TF-IDF ---
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df_seimbang['judul_bersih'])
# Label Encoder
le = LabelEncoder()
y = le.fit_transform(df_seimbang['kategori_utama'])

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, stratify=y, random_state=42)

# --- 4. Implementasi DFWNB ---
# Pilih 100 fitur terbaik berdasarkan chi2
k = 100
chi2_selector = SelectKBest(chi2, k=k)
chi2_selector.fit(X_train, y_train)
# Ambil indeks fitur yang terpilih
selected_indices = chi2_selector.get_support(indices=True)
# Buat bobot fitur: default 1, untuk fitur penting jadi 2
weights = np.ones(X_train.shape[1])
weights[selected_indices] = 2  # DFW: 2 untuk fitur penting, 1 untuk lainnya

# Membuat Fungsi Model DFWNB
from collections import defaultdict
import numpy as np

class DFWNaiveBayes:
    def fit(self, X, y, weights):
        self.classes = np.unique(y)
        self.feature_count = X.shape[1]
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}

        for c in self.classes:
            idx = np.where(y == c)[0]
            X_c = X[idx]
            # prior
            self.class_log_prior_[c] = np.log(len(idx) / len(y))
            # hitung probabilitas kata per kelas dengan bobot fitur
            total = np.asarray((X_c @ weights.reshape(-1, 1)).sum(axis=0)).flatten()[0]
            word_counts = np.asarray((X_c.multiply(weights)) .sum(axis=0)).flatten()
            self.feature_log_prob_[c] = np.log((word_counts + 1) / (total + self.feature_count))  # Laplace smoothing

    def predict(self, X):
        result = []
        for x in X:
            probs = []
            for c in self.classes:
                log_prob = self.class_log_prior_[c] + x.dot(self.feature_log_prob_[c])
                probs.append(log_prob)
            result.append(np.argmax(probs))
        return np.array(result)

# Latih model
dfw_nb = DFWNaiveBayes()
dfw_nb.fit(X_train, y_train, weights)

# --- 5. Evaluasi Model ---
# Prediksi dan evaluasi
y_pred = dfw_nb.predict(X_test)
# Evaluasi performa
from sklearn.metrics import classification_report, accuracy_score
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Tampilkan confusion matrix dengan label kelas
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
plt.figure(figsize=(10, 8))  # opsional, supaya lebih besar
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix - DFW Naive Bayes')
plt.tight_layout()
plt.show()

# --- 6. Simpan Model ---
import joblib
# Simpan model
joblib.dump(nb_model, 'nb_model.pkl')
# Simpan TF-IDF vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
