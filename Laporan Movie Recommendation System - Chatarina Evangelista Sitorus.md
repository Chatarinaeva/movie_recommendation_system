# Laporan Proyek Machine Learning: Movie Recommendation System Using Content-Based and Collaborative Filtering - Chatarina Evangelista Sitorus

## Project Overview

Industri hiburan digital telah mengalami perkembangan pesat dalam dekade terakhir, terutama melalui layanan *streaming* seperti Netflix, Disney+, dan Amazon Prime. Namun, semakin banyaknya pilihan film justru memunculkan tantangan baru bagi pengguna, yakni kesulitan dalam menemukan tontonan yang relevan dan sesuai preferensi. Untuk mengatasi permasalahan ini, sistem rekomendasi hadir sebagai solusi yang efektif dalam menyaring informasi serta meningkatkan pengalaman pengguna secara personal dan efisien.

Proyek ini bertujuan membangun sistem rekomendasi film berbasis *machine learning* menggunakan dua pendekatan utama: *Content-Based Filtering* (CBF) dan *Collaborative Filtering* (CF). CBF memanfaatkan informasi konten seperti genre untuk mengukur kemiripan antar film  [[3]](https://doi.org/10.21108/ijoict.v9i2.747), sedangkan CF menggunakan data historis interaksi pengguna untuk memprediksi preferensi baru berdasarkan kemiripan pola dengan pengguna lain [[1]](https://doi.org/10.1109/ICESC48915.2020.9155879).

Algor dan Srivastava [[1]](https://doi.org/10.1109/ICESC48915.2020.9155879) menunjukkan bahwa *deep learning*-based models dan metrik kemiripan seperti *cosine similarity* sangat efektif dalam sistem CBF, sementara pendekatan CF terbukti efektif dalam mengatasi permasalahan *cold-start item*. Nurhaida dan Marzuki [[3]](https://doi.org/10.21108/ijoict.v9i2.747) mendukung efektivitas *cosine similarity* untuk menganalisis kemiripan konten berbasis genre. Nand dan Tripathi [[4]](https://doi.org/10.12720/jait.12.3.189-196) menambahkan bahwa penggunaan metode hybrid seperti K-Means dan TF-IDF mampu meningkatkan akurasi sistem rekomendasi. Sementara itu, Rukmi *et al.* [[2]](https://doi.org/10.47738/jads.v4i3.115) menekankan bahwa pemilihan algoritma sistem rekomendasi memiliki dampak langsung terhadap kepuasan dan retensi pengguna dalam layanan hiburan digital.

## Business Understanding

Dengan mempertimbangkan latar belakang tersebut, maka proyek ini dirancang dengan tujuan dan strategi sebagai berikut.

### Problem Statements

- Bagaimana membangun sistem rekomendasi film yang mampu menyarankan film serupa berdasarkan genre dari film yang telah disukai pengguna sebelumnya (CBF)?

- Bagaimana merancang model rekomendasi yang dapat memanfaatkan pola rating pengguna lain untuk menyarankan film yang relevan secara personal (CF)?

- Bagaimana mengevaluasi performa kedua pendekatan sistem rekomendasi dan menentukan pendekatan yang paling efektif untuk konteks ini?

### Goals

- Mengembangkan model *Content-Based Filtering* menggunakan representasi TF-IDF pada data genre dan menghitung kemiripan antar film menggunakan cosine similarity.

- Mengembangkan model *Collaborative Filtering* berbasis neural network untuk mempelajari representasi pengguna dan film dalam bentuk *embedding*, lalu memprediksi kemungkinan rating pada film yang belum ditonton.

- Menghasilkan daftar Top-N Recommendation untuk pengguna dan membandingkan kinerja kedua pendekatan berdasarkan metrik evaluasi seperti RMSE.

### Solution statements
Untuk mencapai tujuan di atas, dua pendekatan sistem rekomendasi diterapkan:

- *Content-Based Filtering* (CBF):
Genre dari setiap film akan diolah menggunakan TF-IDF Vectorizer, kemudian dihitung tingkat kemiripannya antar film menggunakan cosine similarity. Sistem akan merekomendasikan film dengan skor kemiripan tertinggi terhadap film yang pernah ditonton pengguna.

- *Collaborative Filtering* (CF):
Model dibangun dengan pendekatan embedding neural network yang mempelajari hubungan laten antara user dan item. Model kemudian memprediksi skor ketertarikan pengguna terhadap film yang belum mereka tonton, lalu memilih rekomendasi berdasarkan skor tertinggi.

## Data Understanding
### **Data Understanding**

Proyek ini menggunakan dataset [*Movie Recommender System*](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset) dari Kaggle. Dataset ini terdiri dari dua file utama:

- `movies.csv` berisi metadata dari setiap film, seperti `movieId`, `title`, dan `genres`.
- `ratings.csv` memuat data interaksi pengguna dengan film, mencakup `userId`, `movieId`, `rating`, dan `timestamp`.

Dataset dimuat ke Google Colab melalui KaggleHub dan disalin ke Google Drive untuk menjaga ketersediaan data secara stabil.

---

### **Exploratory Data Analysis (EDA)**

EDA dilakukan untuk memahami struktur dan karakteristik awal dari `df_movies` dan `df_ratings`. Tahapan ini penting agar potensi masalah seperti duplikasi, nilai kosong, atau inkonsistensi bisa diantisipasi sejak awal.

#### 1. Menampilkan Contoh Data Acak

```python
df_movies.sample(5)
df_ratings.sample(5)
```

## Data Preparation

Proses *data preparation* dilakukan untuk memastikan bahwa data yang digunakan dalam sistem rekomendasi film ini bersih, efisien, dan terstruktur. Tahapan ini dibagi menjadi dua pendekatan utama, yaitu **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**, yang masing-masing membutuhkan preprocessing tersendiri. Dataset juga dibatasi ukurannya agar pelatihan model lebih ringan dan cepat.

### 1. Pembatasan Dataset untuk Efisiensi

**Kode Program:**
```python
subset_movies = df_movies.iloc[:10000]
subset_ratings = df_ratings.iloc[:5000]
```

**Proses yang dilakukan:**  
Mengambil 10.000 data film dan 5.000 data rating dari dataset asli.

**Alasan mengapa diperlukan:**  
Pembatasan ini dilakukan untuk mempercepat proses eksplorasi dan pelatihan awal. Dataset yang lebih kecil memungkinkan eksperimen berjalan lebih cepat, menghemat sumber daya, dan tetap cukup representatif untuk menguji konsep dan validasi awal sistem rekomendasi..


### 2. Preprocessing Content-Based Filtering (CBF)

#### a. Menyalin Data dari Subset Utama

**Kode Program:**
```python
df_movies_cb = subset_movies.copy()
df_ratings_cb = subset_ratings.copy()
```

**Proses yang Dilakukan:**  
Menyalin data film dan rating untuk digunakan dalam preprocessing CBF.

**Alasan mengapa diperlukan:**  
Menyalin data mencegah perubahan tidak disengaja terhadap dataset asli, terutama saat eksplorasi dan transformasi data. Ini juga memastikan bahwa preprocessing CBF tidak memengaruhi data yang digunakan dalam pendekatan lain (misalnya CF), sehingga menjaga modularitas dan kebersihan pipeline.

#### b. Memisahkan Genre Menjadi Baris Terpisah

**Kode Program:**
```python
df_movies_cb = df_movies_cb.assign(genres=df_movies_cb['genres'].str.split('|')).explode('genres')
```

**Proses yang dilakukan:**  
Kolom `genres` yang semula berupa string dipisahkan per genre menggunakan `explode`.

**Alasan mengapa diperlukan:**  
Dengan memecah genre menjadi baris individual, kita dapat menghitung kemiripan antar film berdasarkan genre dengan lebih akurat. Representasi genre per baris memungkinkan penerapan teknik representasi teks seperti TF-IDF, sehingga sistem dapat mengenali kemiripan film berdasarkan kemunculan genre yang sama. Ini krusial untuk pendekatan Content-Based Filtering agar fitur konten film bisa dimanfaatkan secara maksimal.


### 3. Preprocessing Collaborative Filtering (CF)

#### a. Menyalin Data dari Subset Utama

**Kode Program:**
```python
df_movies_cf = subset_movies.copy()
df_ratings_cf = subset_ratings.copy()
```

**Proses yang dilakukan:**  
Menyalin data untuk proses preprocessing Collaborative Filtering.

**Alasan mengapa diperlukan:**  
Penyalinan data memastikan proses preprocessing untuk CF tidak menyebabkan konflik atau modifikasi tak disengaja pada data yang juga digunakan oleh pendekatan lain seperti CBF. Ini menjaga independensi antar pipeline dan memudahkan debugging serta pengujian.

#### b. Encoding `userId` dan `movieId` ke Bentuk Numerik

**Kode Program:**
```python
user_ids_cf = df_ratings_cf['userId'].unique().tolist()
movie_ids_cf = df_ratings_cf['movieId'].unique().tolist()

user_to_index = {uid: idx for idx, uid in enumerate(user_ids_cf)}
index_to_user = {idx: uid for idx, uid in enumerate(user_ids_cf)}

movie_to_index = {mid: idx for idx, mid in enumerate(movie_ids_cf)}
index_to_movie = {idx: mid for idx, mid in enumerate(movie_ids_cf)}

df_ratings_cf['user'] = df_ratings_cf['userId'].map(user_to_index)
df_ratings_cf['movie'] = df_ratings_cf['movieId'].map(movie_to_index)
```

**Proses yang dilakukan:**  
Konversi ID pengguna dan film menjadi indeks numerik.

**Alasan mengapa diperlukan:**  
Model machine learning tidak dapat memproses ID dalam bentuk string atau angka acak sebagai representasi entitas. Dengan mengubahnya ke indeks numerik, model dapat mempelajari hubungan antar entitas secara efisien, dan proses ini juga mempermudah pemetaan ulang ke ID asli jika dibutuhkan.

#### c. Normalisasi Nilai Rating

**Kode Program:**
```python
df_ratings_cf['rating'] = df_ratings_cf['rating'].astype(np.float32)
min_rating = df_ratings_cf['rating'].min()
max_rating = df_ratings_cf['rating'].max()

y_cf = df_ratings_cf['rating'].apply(lambda r: (r - min_rating) / (max_rating - min_rating)).values
```

**Proses yang dilakukan:**  
Mengubah tipe data rating menjadi float dan melakukan normalisasi min-max ke skala 0–1.

**Alasan mengapa diperlukan:**  
Skala rating yang tidak seragam (misalnya dari 0–5) dapat menyebabkan ketidakseimbangan selama proses pelatihan. Normalisasi ke rentang 0–1 membuat pembelajaran lebih stabil, mempercepat konvergensi, dan mengurangi risiko nilai ekstrem memengaruhi bobot model secara tidak proporsional.

#### d. Mengacak dan Membagi Data

**Kode Program:**
```python
df_ratings_cf = df_ratings_cf.sample(frac=1, random_state=42)

x_cf = df_ratings_cf[['user', 'movie']].values

train_indices = int(0.8 * len(x_cf))
x_train_cf, x_val_cf = x_cf[:train_indices], x_cf[train_indices:]
y_train_cf, y_val_cf = y_cf[:train_indices], y_cf[train_indices:]
```

**Proses yang dilakukan:**  
Mengacak dataset dan membaginya menjadi 80% data latih dan 20% data validasi.

**Alasan mengapa diperlukan:**  
Pengacakan data mencegah model belajar dari urutan data yang mungkin memiliki pola tertentu (misalnya berdasarkan waktu). Pembagian data pelatihan dan validasi penting untuk mengevaluasi performa model secara adil, memastikan bahwa model diuji pada data yang belum pernah dilihat selama pelatihan..


#### Ringkasan Tahapan Data Preparation

| No | Tahapan                        | Deskripsi Singkat                                                                 | Alasan Utama                                                                 |
|----|--------------------------------|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 1  | Pembatasan Dataset             | Mengambil sebagian kecil data film dan rating                                    | Efisiensi pelatihan                                                          |
| 2  | Salin Data (CBF & CF)          | Menyalin data dari subset utama                                                  | Isolasi proses dan keamanan data                                             |
| 3  | Genre Explode (CBF)            | Memecah kolom genre menjadi satu genre per baris                                 | Analisis genre secara individual untuk TF-IDF                               |
| 4  | Encoding userId dan movieId (CF)    | Konversi ID pengguna dan film ke indeks numerik                                  | Kompatibilitas dengan model                                                  |
| 5  | Normalisasi Rating  (CF)           | Ubah rating ke skala 0–1                                                         | Skala seragam untuk pelatihan stabil                                         |
| 6  | Shuffle & Split Train-Validation (CF) | Mengacak data dan membagi 80% train, 20% validasi                             | Evaluasi model yang objektif                                                 |


#### Insight Data Preparation

Setiap langkah dalam proses data preparation dirancang untuk memastikan data berada dalam kondisi terbaik sebelum digunakan dalam tahap pemodelan. Penggunaan pendekatan manual, seperti pada proses encoding, normalisasi, dan pembagian data, memberikan tingkat kendali dan fleksibilitas yang lebih tinggi dalam menyesuaikan data terhadap kebutuhan model.

Persiapan data yang cermat memainkan peran penting dalam kesuksesan sistem rekomendasi, karena data yang terstruktur dengan baik akan meningkatkan akurasi prediksi dan relevansi hasil rekomendasi bagi pengguna.


---

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

## **Kesimpulan**

Sistem rekomendasi film berhasil dibangun dengan dua pendekatan utama: *Content-Based Filtering* (CBF) dan *Collaborative Filtering* (CF).

- Pendekatan CBF menggunakan genre film untuk menghitung kemiripan antar film dan berhasil memberikan rekomendasi yang konsisten berdasarkan kategori.

- Pendekatan CF dilatih menggunakan data rating pengguna dan menghasilkan validation RMSE sebesar 0.2324, menunjukkan performa yang baik dan stabil.

Kombinasi kedua pendekatan ini memberikan solusi yang saling melengkapi: CBF berguna untuk pengguna baru (tanpa riwayat rating), sedangkan CF sangat efektif untuk memberikan rekomendasi yang dipersonalisasi.

Dengan demikian, sistem ini mampu memberikan Top-N recommendation yang akurat, relevan, dan adaptif terhadap berbagai kebutuhan pengguna.

## **Referensi**

[1] S. Algor and S. Srivastava, “Hybrid Movie Recommendation System using Content-Based and Collaborative Filtering,” in *Proc. 2020 Int. Conf. Electronics and Sustainable Communication Systems (ICESC)*, 2020, pp. 102–106. doi: [10.1109/ICESC48915.2020.9155879](https://doi.org/10.1109/ICESC48915.2020.9155879)

[2] A. R. Rukmi, F. A. Permana, and D. E. Maharsi, “Hybrid Recommendation System for Movie Selection using TF-IDF and Neural CF,” *J. Appl. Data Sci.*, vol. 4, no. 3, pp. 211–221, 2022. doi: [10.47738/jads.v4i3.115](https://doi.org/10.47738/jads.v4i3.115)

[3] I. Nurhaida and M. Marzuki, “Movie Recommendation System using Content-Based Filtering and Cosine Similarity,” *Indones. J. Inf. Commun. Technol.*, vol. 9, no. 2, pp. 101–110, 2021. doi: [10.21108/ijoict.v9i2.747](https://doi.org/10.21108/ijoict.v9i2.747)

[4] K. Nand and R. Tripathi, “Movie Recommendation System Based on Hybrid Filtering using K-Means and TF-IDF,” *J. Adv. Inf. Technol.*, vol. 12, no. 3, pp. 189–196, 2021. doi: [10.12720/jait.12.3.189-196](https://doi.org/10.12720/jait.12.3.189-196)

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
