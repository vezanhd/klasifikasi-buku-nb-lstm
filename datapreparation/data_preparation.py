from google.colab import files
uploaded = files.upload()

----- Load Data -------
import pandas as pd
# Membaca file
df = pd.read_excel("Data Awal.xlsx", header=1)

----- Pemilihan Fitur -----
# Ambil kolom yang dibutuhkan
df = df[['urut', 'JUDUL UTAMA', 'ANAK JUDUL', 'NO DDC', 'BAHASA']]
# Gabungkan kolom judul utama dan anak judul
df['judul'] = df['JUDUL UTAMA'].fillna('').astype(str) + ' ' + df['ANAK JUDUL'].fillna('').astype(str)
# Sesuaikan kolom agar lebih rapi
df = df[['judul', 'NO DDC', 'BAHASA']]
df = df.rename(columns={'NO DDC': 'kategori', 'BAHASA': 'bahasa'})

----- Cleaning Data-----
# Menghapus data duplikat dari dataframe utama
# keep='first' artinya kemunculan pertama dianggap AMAN, kemunculan kedua dst dianggap DUPLIKAT
df = df.drop_duplicates(subset='judul', keep='first')

# Menghapus data missing berdasarkan kolom judul atau kategori
df = df.dropna(subset=['judul', 'kategori'])

# Menghapus data kategori invalid (Tidak Sesuai No DDC)
# Coba deteksi mana yang akan error jika dikonversi ke angka
# Gunakan variabel sementara 'cek_numeric' agar data asli di 'df' tidak berubah dulu
cek_numeric = pd.to_numeric(df['kategori'], errors='coerce')
# Identifikasi baris yang bernilai NaN (artinya kategori aslinya bukan angka valid)
mask_invalid = cek_numeric.isna()
df_invalid_kategori = df[mask_invalid]
# Ambil baris yang valid saja (kebalikan dari mask_invalid)
df = df[~mask_invalid].copy()
# Konversi kolom kategori ke integer
df['kategori'] = pd.to_numeric(df['kategori']).astype(int)

# Konversi Kategori Menjadi 10 Kategori Utama DDC
# Buat kolom kategori_utama berdasarkan pembulatan ke bawah (kelipatan 100)
df['kategori_utama'] = (df['kategori'] // 100) * 100
# Ubah ke format string 3 digit agar rapi (misal: 100 → '100', 0 → '000')
df['kategori_utama'] = df['kategori_utama'].astype(str).str.zfill(3)

# Ambil Kolom yang diperlukan Untuk Step Selanjutnya 
df = df[['judul', 'kategori_utama','bahasa']]
# Cek hasil akhir
print("Jumlah data:", df.shape[0])
df.head()
