# Dashboard Analisis E-Commerce Olist

Proyek analisis data pada E-Commerce Public Dataset (marketplace Olist, Brasil) untuk periode analisis 1 Januari 2017–31 Agustus 2018. Proyek ini menjawab tiga pertanyaan bisnis:

1. Bagaimana pertumbuhan nilai penjualan item dan jumlah pesanan pada Januari–Agustus 2018 dibandingkan Januari–Agustus 2017, kategori mana yang memberi kenaikan terbesar, dan bagaimana tren bulanan 2018?
2. Bagaimana perbedaan rata-rata review score antara pesanan tepat waktu dan terlambat, dan di negara bagian mana persentase pesanan terlambat paling tinggi?
3. Berapa persen pelanggan yang memiliki lebih dari satu pesanan dalam periode analisis, dan segmen RFM mana yang menyumbang nilai penjualan item terbesar?

Analisis lanjutan meliputi RFM analysis dengan pengelompokan manual, geospatial analysis (peta dan jarak haversine), serta binning. Tidak ada algoritma machine learning.

## Struktur proyek

```
submission/
├── dashboard/
│   ├── dashboard.py        # aplikasi Streamlit
│   └── main_data.csv       # data hasil olahan notebook, dibaca dashboard
├── data/                   # sembilan berkas CSV sumber
│   ├── customers_dataset
│   ├── geolocation_dataset
│   ├── order_items_dataset
│   ├── order_payments_dataset
│   ├── order_reviews_dataset
│   ├── orders_dataset
│   ├── product_category_name_translation
│   ├── products_dataset
│   ├── sellers_dataset
├── notebook.ipynb          # proses analisis dan validasi
├── README.md
├── requirements.txt
└── url.txt                 # URL dashboard setelah deployment

```

# Setup Environment ✨

## Setup Environment — Anaconda
```
conda create --name olist-analytics python=3.12 -y
conda activate olist-analytics
pip install -r requirements.txt
```

## Setup Environment — Shell/Terminal
```
Pastikan Python 3.12 dan pip sudah terpasang.

    mkdir olist-ecommerce-analytics
    cd olist-ecommerce-analytics

Tempatkan file proyek beserta requirements.txt ke dalam direktori proyek, lalu jalankan:

    python -m venv .venv

Aktifkan virtual environment.

Windows:

    .venv\Scripts\activate

macOS/Linux:

    source .venv/bin/activate

Instal dependensi:

    pip install -r requirements.txt
```

## Run Streamlit App
```
Dari root direktori proyek, jalankan:

    streamlit run dashboard/dashboard.py
```

## Menjalankan ulang notebook
```
Buka `notebook.ipynb` dengan Jupyter, VS Code, atau Google Colab dari root proyek, lalu jalankan seluruh sel secara berurutan. Notebook membaca `data/*.csv` menggunakan path relatif terhadap root proyek dan menulis `dashboard/main_data.csv`.

```
## Isi dashboard

| Tab | Isi |
|---|---|
| Penjualan dan kategori | Perbandingan year-over-year Jan–Agu 2018 dan 2017, tren bulanan, kategori teratas, kontribusi kenaikan per kategori |
| Pengiriman dan review | Review score menurut kelompok keterlambatan, persentase terlambat per negara bagian, tren bulanan |
| Peta dan jarak | Peta bubble per negara bagian, jarak penjual–pelanggan, rute dalam dan lintas negara bagian |
| Segmentasi pelanggan (RFM) | Distribusi frekuensi pembelian, porsi pelanggan dan nilai penjualan per segmen, aturan segmentasi |
| Data dan definisi | Definisi metrik dan cuplikan data |

Filter di sisi kiri: rentang tanggal pembelian dan negara bagian pelanggan. Perbandingan year-over-year dan segmentasi RFM menggunakan periode analisis penuh dan hanya mengikuti filter negara bagian.

## Definisi metrik dan batas interpretasi

- **Pesanan valid untuk analisis:** status `delivered` dan tanggal tiba di pelanggan terisi; analisis dibatasi pada pesanan dengan tanggal pembelian 1 Januari 2017–31 Agustus 2018.
- **Nilai penjualan item (disebut `revenue` dalam beberapa grafik):** jumlah `price` semua item pada pesanan valid, tidak termasuk ongkos kirim, biaya pembayaran, biaya operasional, atau retur. Ini bukan laba bersih maupun total pembayaran. Satuan R$ (real Brasil).
- **Pesanan terlambat:** tanggal tiba di pelanggan, dibandingkan pada tingkat hari kalender, lebih besar daripada tanggal estimasi.
- **Pelanggan:** `customer_unique_id`.
- **RFM:** recency dihitung hingga 1 September 2018; segmen dibentuk dengan aturan bisnis eksplisit (aktif bila recency 180 hari atau kurang, nilai tinggi bila pengeluaran berada di persentil ke-80 atau lebih).
- **Kohort pembelian ulang:** “pesanan pertama” berarti pesanan pertama yang teramati di periode analisis, bukan selalu pembelian pertama sepanjang riwayat pelanggan. Tingkat pembelian ulang 180 hari hanya dihitung pada pelanggan yang pesanan pertama teramatinya diterima setidaknya 180 hari sebelum tanggal acuan.
- **Interpretasi keterlambatan dan review:** perbedaan skor antara pesanan tepat waktu dan terlambat adalah asosiasi deskriptif, bukan bukti kausal bahwa keterlambatan saja menyebabkan skor lebih rendah.
- **Anomali review ID:** sebagian `review_id` terhubung ke lebih dari satu `order_id`. Karena sumber tidak menyediakan aturan pemetaan otoritatif, ID tersebut tidak dideduplikasi secara arbitrer; analisis memilih ulasan terbaru per pesanan secara deterministik. Pemeriksaan sensitivitas menunjukkan perubahan kecil pada metrik agregat, tetapi anomali tetap dicatat sebagai keterbatasan data.
