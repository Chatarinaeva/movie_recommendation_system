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
Paragraf awal bagian ini menjelaskan informasi mengenai jumlah data, kondisi data, dan informasi mengenai data yang digunakan. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya, uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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
