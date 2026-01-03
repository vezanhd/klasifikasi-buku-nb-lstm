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
