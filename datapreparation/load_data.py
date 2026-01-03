from google.colab import files
uploaded = files.upload()

import pandas as pd
# Membaca file
df = pd.read_excel("Data Awal.xlsx", header=1)
