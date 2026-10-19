"""Dashboard analisis e-commerce Olist (pesanan delivered, Januari 2017 - Agustus 2018).

Sumber data: main_data.csv (hasil pengolahan notebook.ipynb) pada folder yang sama dengan berkas ini.
Jalankan dari root proyek:  streamlit run dashboard/dashboard.py
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Dashboard E-Commerce Olist", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "main_data.csv"

# ----------------------------------------------------------------------------
# Konstanta (nilai yang sama dengan notebook.ipynb)
# ----------------------------------------------------------------------------
BIRU, BIRU_MUDA, ORANYE, ABU, GELAP = "#2B6CB0", "#A9C4E4", "#DD6B20", "#A0AEC0", "#2D3748"
BULAN_ID = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

TANGGAL_ACUAN_RFM = pd.Timestamp("2018-09-01")
BATAS_RECENCY_HARI = 180
MIN_PESANAN_STATE = 100          # negara bagian dengan pesanan lebih sedikit tidak diperingkat
BASIS_MINIMAL = 20_000           # pertumbuhan % kategori hanya dihitung bila revenue Jan-Agu 2017 >= nilai ini

BATAS_DELAY = [-np.inf, -8, 0, 3, 7, 14, np.inf]
LABEL_DELAY = ["Lebih awal >7 hari", "Tepat waktu / awal ≤7 hari", "Terlambat 1–3 hari",
               "Terlambat 4–7 hari", "Terlambat 8–14 hari", "Terlambat >14 hari"]
BATAS_JARAK = [-0.1, 100, 500, 1000, 2000, np.inf]
LABEL_JARAK = ["≤100 km", "101–500 km", "501–1.000 km", "1.001–2.000 km", ">2.000 km"]
URUTAN_SEGMEN = ["Loyal - Aktif", "Loyal - Tidak Aktif", "Baru - Nilai Tinggi", "Baru - Reguler",
                 "Dormant - Nilai Tinggi", "Dormant - Reguler"]

_PARAMETER_PLOTLY = inspect.signature(st.plotly_chart).parameters


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
def angka(x, desimal=0):
    """Format angka gaya Indonesia: titik sebagai pemisah ribuan, koma sebagai desimal."""
    teks = f"{x:,.{desimal}f}"
    return teks.replace(",", "_").replace(".", ",").replace("_", ".")


def nama_kategori(kode):
    return kode.replace("_", " ").title()


def label_bulan(tanggal):
    return f"{BULAN_ID[tanggal.month - 1]} {tanggal.year}"


def tampilkan(fig):
    """Tampilkan grafik Plotly selebar kontainer (kompatibel dengan versi Streamlit lama dan baru)."""
    if "width" in _PARAMETER_PLOTLY:
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None)


def gaya(fig, tinggi=420, legenda=False):
    fig.update_layout(
        template="plotly_white", height=tinggi, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color=GELAP, size=13),
        separators=",.", showlegend=legenda,
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E0")
    fig.update_yaxes(gridcolor="#E2E8F0", zerolinecolor="#CBD5E0")
    return fig


def persen_skor_rendah(skor):
    """Persentase ulasan bernilai 1-2 dari pesanan yang memiliki ulasan."""
    ada = skor.dropna()
    return (ada <= 2).mean() * 100 if len(ada) else np.nan


def batang_horizontal(label, nilai, teks, warna, judul_sumbu, tinggi=None, latar_label=False):
    """Batang horizontal dengan label nilai di ujung batang.

    latar_label=True menggambar label sebagai anotasi berlatar putih supaya garis acuan tidak menimpa teks.
    """
    fig = go.Figure(go.Bar(
        x=nilai, y=label, orientation="h", marker_color=warna, customdata=teks,
        text=None if latar_label else teks, textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{customdata}<extra></extra>",
    ))
    if latar_label:
        for lab, nilai_bar, teks_bar in zip(label, nilai, teks):
            fig.add_annotation(x=nilai_bar, y=lab, text=teks_bar, showarrow=False, xanchor="left", xshift=4,
                               bgcolor="white", borderpad=1, font=dict(size=12, color=GELAP))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=judul_sumbu, range=[0, max(nilai) * 1.4 if len(nilai) else 1], tickformat=",.0f")
    return gaya(fig, tinggi or max(300, 34 * len(label) + 80))


# ----------------------------------------------------------------------------
# Pemuatan data dan RFM
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat data...")
def muat_data(path):
    df = pd.read_csv(path, parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                        "order_estimated_delivery_date"])
    df["tanggal_beli"] = df["order_purchase_timestamp"].dt.normalize()
    df["bulan"] = df["tanggal_beli"].dt.to_period("M").dt.to_timestamp()
    return df


def hitung_rfm(data, tanggal_acuan):
    """RFM per pelanggan dari data berbaris item pesanan (logika sama dengan notebook)."""
    rfm = data.groupby("customer_unique_id").agg(
        pembelian_terakhir=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("price", "sum"),
    )
    rfm["recency"] = (tanggal_acuan - rfm["pembelian_terakhir"].dt.normalize()).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[["recency", "frequency", "monetary"]]


def segmentasi_rfm(rfm, batas_recency, batas_monetary):
    """Pengelompokan manual berdasarkan aturan bisnis (tanpa algoritma machine learning)."""
    aktif = rfm["recency"] <= batas_recency
    bernilai_tinggi = rfm["monetary"] >= batas_monetary
    repeat = rfm["frequency"] >= 2
    return pd.Series(np.select(
        [repeat & aktif, repeat & ~aktif, ~repeat & aktif & bernilai_tinggi, ~repeat & aktif, ~repeat & bernilai_tinggi],
        URUTAN_SEGMEN[:5],
        default=URUTAN_SEGMEN[5],
    ), index=rfm.index, name="segmen")


if not DATA_PATH.exists():
    st.error(f"Berkas data tidak ditemukan: {DATA_PATH}. Jalankan notebook.ipynb untuk membuat main_data.csv.")
    st.stop()

df = muat_data(str(DATA_PATH))
rfm_semua = hitung_rfm(df, TANGGAL_ACUAN_RFM)
batas_monetary = round(float(rfm_semua["monetary"].quantile(0.80)), 2)

# ----------------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------------
st.sidebar.header("Filter")
tanggal_min, tanggal_maks = df["tanggal_beli"].min().date(), df["tanggal_beli"].max().date()
rentang = st.sidebar.date_input("Rentang tanggal pembelian", value=(tanggal_min, tanggal_maks),
                                min_value=tanggal_min, max_value=tanggal_maks)
semua_negara = sorted(df["customer_state"].unique())
negara_dipilih = st.sidebar.multiselect("Negara bagian pelanggan", options=semua_negara, default=semua_negara)
st.sidebar.caption("Filter tanggal berlaku pada ringkasan, grafik tren, kategori, pengiriman, dan peta. "
                   "Perbandingan year-over-year dan segmentasi RFM memakai seluruh periode analisis; "
                   "keduanya hanya mengikuti filter negara bagian.")

if not (isinstance(rentang, (tuple, list)) and len(rentang) == 2):
    st.info("Pilih tanggal awal dan tanggal akhir pada filter di sisi kiri.")
    st.stop()
if not negara_dipilih:
    st.warning("Pilih minimal satu negara bagian pada filter di sisi kiri.")
    st.stop()

tanggal_awal, tanggal_akhir = pd.Timestamp(rentang[0]), pd.Timestamp(rentang[1])
dasar = df[df["customer_state"].isin(negara_dipilih)]                              # filter negara bagian saja
data = dasar[dasar["tanggal_beli"].between(tanggal_awal, tanggal_akhir)]          # filter tanggal + negara bagian
if data.empty:
    st.warning("Tidak ada pesanan pada kombinasi filter yang dipilih.")
    st.stop()
pesanan = data.drop_duplicates("order_id")                                          # satu baris per pesanan

# ----------------------------------------------------------------------------
# Judul dan ringkasan
# ----------------------------------------------------------------------------
st.title("Dashboard Analisis E-Commerce Olist")
st.caption("Pesanan berstatus delivered, 1 Januari 2017 sampai 31 Agustus 2018. Nilai dalam real Brasil (R$). "
           "Revenue adalah jumlah harga barang, tidak termasuk ongkos kirim.")

revenue_total = data["price"].sum()
jumlah_pesanan = len(pesanan)
nilai_per_pesanan = revenue_total / jumlah_pesanan
persen_terlambat = pesanan["is_late"].mean() * 100
rata_review = pesanan["review_score"].mean()

kolom = st.columns(5)
kolom[0].metric("Revenue", f"R$ {angka(revenue_total / 1e6, 2)} juta")
kolom[1].metric("Pesanan", angka(jumlah_pesanan))
kolom[2].metric("Nilai per pesanan", f"R$ {angka(nilai_per_pesanan, 2)}")
kolom[3].metric("Rata-rata review score", angka(rata_review, 2) if pd.notna(rata_review) else "-")
kolom[4].metric("Pesanan terlambat", f"{angka(persen_terlambat, 1)}%",
                help="Tanggal tiba (per hari kalender) melewati tanggal estimasi pengiriman.")

def jawaban_singkat():
    """Ringkasan jawaban tiga pertanyaan bisnis, dihitung dari seluruh data tanpa filter."""
    jan_agu_semua = df[df["order_purchase_timestamp"].dt.month <= 8]
    per_th = jan_agu_semua.groupby(jan_agu_semua["order_purchase_timestamp"].dt.year).agg(
        revenue=("price", "sum"), pesanan=("order_id", "nunique"))
    yoy_revenue = (per_th.loc[2018, "revenue"] / per_th.loc[2017, "revenue"] - 1) * 100
    yoy_pesanan = (per_th.loc[2018, "pesanan"] / per_th.loc[2017, "pesanan"] - 1) * 100
    rev_bulan = df.groupby("bulan")["price"].sum()
    rev_2018 = rev_bulan[rev_bulan.index.year == 2018]
    puncak_2018 = rev_2018.idxmax()
    turun = (1 - rev_2018.iloc[-1] / rev_2018.max()) * 100
    pv = jan_agu_semua.pivot_table(index="product_category", columns=jan_agu_semua["order_purchase_timestamp"].dt.year,
                                   values="price", aggfunc="sum", fill_value=0)
    kenaikan = (pv[2018] - pv[2017]).sort_values(ascending=False)
    porsi_dua = kenaikan.iloc[:2].sum() / kenaikan.sum() * 100

    ps = df.drop_duplicates("order_id")
    skor = ps.groupby("is_late")["review_score"].mean()
    terlambat_state = ps.groupby("customer_state").agg(n=("order_id", "size"), t=("is_late", "mean"))
    terlambat_state = terlambat_state[terlambat_state["n"] >= MIN_PESANAN_STATE].sort_values("t", ascending=False)
    lima = ", ".join(f"{k} ({angka(v * 100, 1)}%)" for k, v in terlambat_state["t"].head(5).items())

    repeat = (rfm_semua["frequency"] >= 2).mean() * 100
    segmen = segmentasi_rfm(rfm_semua, BATAS_RECENCY_HARI, batas_monetary)
    tinggi = segmen.isin(["Baru - Nilai Tinggi", "Dormant - Nilai Tinggi"])
    pelanggan_tinggi = tinggi.mean() * 100
    revenue_tinggi = rfm_semua.loc[tinggi, "monetary"].sum() / rfm_semua["monetary"].sum() * 100
    return (
        f"**Pertanyaan 1.** Revenue Jan–Agu 2018 tumbuh {angka(yoy_revenue, 1)}% dan pesanan {angka(yoy_pesanan, 1)}% dibanding "
        f"Jan–Agu 2017. Revenue bulanan 2018 tidak lagi naik: Agustus {angka(turun, 1)}% di bawah puncak "
        f"{label_bulan(puncak_2018)}. {nama_kategori(kenaikan.index[0])} dan {nama_kategori(kenaikan.index[1])} menyumbang "
        f"{angka(porsi_dua, 1)}% dari total kenaikan.\n\n"
        f"**Pertanyaan 2.** Review score rata-rata pesanan terlambat {angka(skor[True], 2)}, pesanan tepat waktu {angka(skor[False], 2)}. "
        f"Pesanan terlambat nasional {angka(ps['is_late'].mean() * 100, 1)}%; tertinggi di {lima}.\n\n"
        f"**Pertanyaan 3.** Hanya {angka(repeat, 2)}% pelanggan membeli lebih dari satu kali. Dua segmen bernilai tinggi (Baru dan "
        f"Dormant) mencakup {angka(pelanggan_tinggi, 1)}% pelanggan dan menyumbang {angka(revenue_tinggi, 1)}% revenue."
    )


with st.expander("Jawaban singkat atas tiga pertanyaan bisnis (seluruh periode, tanpa filter)"):
    st.markdown(jawaban_singkat())

tab_penjualan, tab_kirim, tab_peta, tab_rfm, tab_data = st.tabs([
    "Penjualan dan kategori", "Pengiriman dan review", "Peta dan jarak", "Segmentasi pelanggan (RFM)", "Data dan definisi",
])

# ----------------------------------------------------------------------------
# Tab 1: penjualan dan kategori (Pertanyaan 1)
# ----------------------------------------------------------------------------
with tab_penjualan:
    st.subheader("Pertumbuhan penjualan dan kategori penyumbang kenaikan")

    jan_agu = dasar[dasar["order_purchase_timestamp"].dt.month <= 8]
    tahun_beli = jan_agu["order_purchase_timestamp"].dt.year
    per_tahun = jan_agu.groupby(tahun_beli).agg(revenue=("price", "sum"), pesanan=("order_id", "nunique"))

    st.markdown("**Perbandingan Januari–Agustus 2018 terhadap Januari–Agustus 2017** (mengikuti filter negara bagian)")
    if {2017, 2018}.issubset(per_tahun.index):
        r17, r18 = per_tahun.loc[2017, "revenue"], per_tahun.loc[2018, "revenue"]
        p17, p18 = per_tahun.loc[2017, "pesanan"], per_tahun.loc[2018, "pesanan"]
        a17, a18 = r17 / p17, r18 / p18

        def delta(sekarang, sebelum):
            persen = (sekarang / sebelum - 1) * 100
            return f"{'+' if persen >= 0 else '-'}{angka(abs(persen), 1)}% dari Jan–Agu 2017"

        k1, k2, k3 = st.columns(3)
        k1.metric("Revenue Jan–Agu 2018", f"R$ {angka(r18 / 1e6, 2)} juta", delta(r18, r17))
        k2.metric("Pesanan Jan–Agu 2018", angka(p18), delta(p18, p17))
        k3.metric("Nilai per pesanan Jan–Agu 2018", f"R$ {angka(a18, 2)}", delta(a18, a17))
        st.caption("Pertumbuhan berasal dari jumlah pesanan; nilai per pesanan nyaris tidak berubah. "
                   "Basis 2017 masih kecil, sehingga persentase pertumbuhan melebih-lebihkan momentum terkini.")
    else:
        st.info("Perbandingan year-over-year memerlukan pesanan pada Januari–Agustus 2017 dan 2018 untuk negara bagian terpilih.")

    st.markdown("---")
    st.markdown("**Tren bulanan**")
    metrik_tren = st.radio("Metrik", ["Revenue (R$ ribu)", "Jumlah pesanan"], horizontal=True, key="metrik_tren")
    bulanan = data.groupby("bulan").agg(revenue=("price", "sum"), pesanan=("order_id", "nunique"))
    if metrik_tren.startswith("Revenue"):
        nilai_bulanan, satuan = bulanan["revenue"] / 1e3, " rb"
    else:
        nilai_bulanan, satuan = bulanan["pesanan"], ""
    teks_hover = [f"{label_bulan(t)}: {'R$ ' if satuan else ''}{angka(v)}{satuan}" for t, v in nilai_bulanan.items()]
    fig = go.Figure(go.Scatter(x=nilai_bulanan.index, y=nilai_bulanan.values, mode="lines+markers",
                               line=dict(color=BIRU, width=3), marker=dict(size=7), customdata=teks_hover,
                               hovertemplate="%{customdata}<extra></extra>"))
    puncak = nilai_bulanan.idxmax()
    fig.add_annotation(x=puncak, y=nilai_bulanan.max(), text=f"Puncak: {label_bulan(puncak)}", showarrow=True,
                       arrowhead=0, ax=0, ay=-32, font=dict(size=12))
    fig.update_xaxes(tickmode="array", tickvals=list(nilai_bulanan.index),
                     ticktext=[label_bulan(t) for t in nilai_bulanan.index], tickangle=-45)
    fig.update_yaxes(title=metrik_tren, rangemode="tozero", range=[0, nilai_bulanan.max() * 1.18], tickformat=",.0f")
    tampilkan(gaya(fig, 400))

    st.markdown("---")
    st.markdown("**Kategori dengan revenue terbesar pada rentang tanggal terpilih**")
    jumlah_kategori = st.slider("Jumlah kategori ditampilkan", min_value=5, max_value=15, value=10, key="n_kategori")
    revenue_kategori = data.groupby("product_category")["price"].sum().sort_values(ascending=False)
    teratas = revenue_kategori.head(jumlah_kategori)
    porsi = teratas / revenue_kategori.sum() * 100
    fig = batang_horizontal(
        [nama_kategori(k) for k in teratas.index], (teratas / 1e3).values,
        [f"R$ {angka(v / 1e3)} rb ({angka(p, 1)}%)" for v, p in zip(teratas, porsi)],
        [BIRU if i < 3 else BIRU_MUDA for i in range(len(teratas))], "Revenue (R$ ribu)")
    tampilkan(fig)
    st.caption(f"{jumlah_kategori} kategori teratas menyumbang {angka(porsi.sum(), 1)}% revenue pada rentang terpilih.")

    st.markdown("---")
    st.markdown("**Kontribusi kenaikan revenue per kategori (Jan–Agu 2018 dibandingkan Jan–Agu 2017)**")
    if {2017, 2018}.issubset(per_tahun.index):
        pivot = jan_agu.pivot_table(index="product_category", columns=tahun_beli, values="price",
                                    aggfunc="sum", fill_value=0)
        pivot.columns.name = None
        kategori_yoy = pd.DataFrame({"revenue_2017": pivot[2017], "revenue_2018": pivot[2018]})
        kategori_yoy["kenaikan"] = kategori_yoy["revenue_2018"] - kategori_yoy["revenue_2017"]
        total_kenaikan = kategori_yoy["kenaikan"].sum()
        naik = kategori_yoy.sort_values("kenaikan", ascending=False).head(10)
        label_naik, teks_naik = [], []
        for kode, baris in naik.iterrows():
            tumbuh = (f"+{angka(baris['kenaikan'] / baris['revenue_2017'] * 100)}%"
                      if baris["revenue_2017"] >= BASIS_MINIMAL else "basis 2017 kecil")
            label_naik.append(nama_kategori(kode))
            teks_naik.append(f"+R$ {angka(baris['kenaikan'] / 1e3)} rb ({tumbuh})")
        fig = batang_horizontal(label_naik, (naik["kenaikan"] / 1e3).values, teks_naik,
                                [BIRU if i < 2 else BIRU_MUDA for i in range(len(naik))],
                                "Kenaikan revenue (R$ ribu)")
        tampilkan(fig)
        if total_kenaikan > 0:
            st.caption(f"Sepuluh kategori menyumbang {angka(naik['kenaikan'].sum() / total_kenaikan * 100, 1)}% dari total "
                       f"kenaikan R$ {angka(total_kenaikan / 1e6, 2)} juta. Persentase pertumbuhan ditampilkan bila "
                       f"revenue Jan–Agu 2017 minimal R$ {angka(BASIS_MINIMAL)}.")
    else:
        st.info("Data Januari–Agustus 2017 dan 2018 diperlukan untuk grafik ini.")

# ----------------------------------------------------------------------------
# Tab 2: pengiriman dan review (Pertanyaan 2)
# ----------------------------------------------------------------------------
with tab_kirim:
    st.subheader("Keterlambatan pengiriman, review score, dan sebarannya per negara bagian")

    status = pesanan.groupby("is_late").agg(pesanan=("order_id", "size"), rata_review=("review_score", "mean"),
                                            skor_rendah=("review_score", persen_skor_rendah))
    skor_tepat = status["rata_review"].get(False, np.nan)
    skor_telat = status["rata_review"].get(True, np.nan)
    rendah_tepat = status["skor_rendah"].get(False, np.nan)
    rendah_telat = status["skor_rendah"].get(True, np.nan)

    k1, k2, k3 = st.columns(3)
    k1.metric("Review score pesanan tepat waktu", angka(skor_tepat, 2) if pd.notna(skor_tepat) else "-")
    k2.metric("Review score pesanan terlambat", angka(skor_telat, 2) if pd.notna(skor_telat) else "-")
    k3.metric("Ulasan 1–2 bintang (terlambat vs tepat waktu)",
              f"{angka(rendah_telat, 1)}% vs {angka(rendah_tepat, 1)}%" if pd.notna(rendah_telat) and pd.notna(rendah_tepat) else "-")

    st.markdown("---")
    st.markdown("**Review score menurut kelompok keterlambatan**")
    kelompok = pd.cut(pesanan["delay_days"], bins=BATAS_DELAY, labels=LABEL_DELAY)
    per_kelompok = pesanan.groupby(kelompok, observed=False).agg(
        pesanan=("order_id", "size"), rata_review=("review_score", "mean"), skor_rendah=("review_score", persen_skor_rendah))
    per_kelompok = per_kelompok[per_kelompok["pesanan"] > 0]
    teks_kelompok = [f"{angka(r, 2)}" for r in per_kelompok["rata_review"]]
    fig = go.Figure(go.Bar(
        x=[f"{k}<br>n = {angka(n)}<br>skor 1–2: {angka(s, 0)}%" for k, n, s in
           zip(per_kelompok.index.astype(str), per_kelompok["pesanan"], per_kelompok["skor_rendah"])],
        y=per_kelompok["rata_review"], text=teks_kelompok, textposition="outside",
        marker_color=[ORANYE if "Terlambat" in str(k) else BIRU for k in per_kelompok.index],
        hovertemplate="%{x}<br>Rata-rata review: %{text}<extra></extra>", cliponaxis=False))
    fig.update_yaxes(title="Rata-rata review score (skala 1–5)", range=[0, 5.4])
    tampilkan(gaya(fig, 440))
    st.caption("Sumbu dimulai dari nol. Terlambat = tanggal tiba (per hari kalender) melewati tanggal estimasi. "
               "Skor 1–2 dihitung dari pesanan yang memiliki ulasan. Perbedaan skor adalah asosiasi deskriptif, "
               "bukan bukti bahwa keterlambatan sendirian menyebabkan skor lebih rendah.")

    st.markdown("---")
    st.markdown("**Persentase pesanan terlambat per negara bagian**")
    per_state = pesanan.groupby("customer_state").agg(pesanan=("order_id", "size"), terlambat=("is_late", "mean"))
    per_state["terlambat"] *= 100
    dibandingkan = per_state[per_state["pesanan"] >= MIN_PESANAN_STATE].sort_values("terlambat", ascending=False)
    diabaikan = per_state[per_state["pesanan"] < MIN_PESANAN_STATE].index.tolist()
    if len(dibandingkan):
        lima_teratas = set(dibandingkan.index[:5])
        fig = batang_horizontal(
            list(dibandingkan.index), dibandingkan["terlambat"].values,
            [f"{angka(t, 1)}% (n = {angka(n)})" for t, n in zip(dibandingkan["terlambat"], dibandingkan["pesanan"])],
            [ORANYE if s in lima_teratas else ABU for s in dibandingkan.index], "Pesanan terlambat (%)",
            tinggi=max(320, 26 * len(dibandingkan) + 90), latar_label=True)
        fig.add_vline(x=persen_terlambat, line_dash="dash", line_color=GELAP,
                      annotation_text=f"Rata-rata terpilih {angka(persen_terlambat, 1)}%", annotation_position="bottom right")
        tampilkan(fig)
        catatan = "Lima negara bagian tertinggi diberi warna oranye. n = jumlah pesanan delivered."
        if diabaikan:
            catatan += f" Tidak ditampilkan karena kurang dari {MIN_PESANAN_STATE} pesanan: {', '.join(sorted(diabaikan))}."
        st.caption(catatan)
    else:
        st.info(f"Tidak ada negara bagian dengan minimal {MIN_PESANAN_STATE} pesanan pada filter ini.")

    st.markdown("---")
    st.markdown("**Tren bulanan keterlambatan dan ulasan 1–2 bintang**")
    kirim_bulanan = pesanan.groupby("bulan").agg(pesanan=("order_id", "size"), terlambat=("is_late", "mean"),
                                                 skor_rendah=("review_score", persen_skor_rendah))
    kirim_bulanan["terlambat"] *= 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Pesanan terlambat (%)", "Ulasan skor 1–2 (%)"))
    teks_bulan = [label_bulan(t) for t in kirim_bulanan.index]
    for baris, kolom_data, warna in [(1, "terlambat", ORANYE), (2, "skor_rendah", BIRU)]:
        fig.add_trace(go.Scatter(x=kirim_bulanan.index, y=kirim_bulanan[kolom_data], mode="lines+markers",
                                 line=dict(color=warna, width=3), marker=dict(size=6), customdata=teks_bulan,
                                 hovertemplate="%{customdata}: %{y:.1f}%<extra></extra>"), row=baris, col=1)
        fig.update_yaxes(range=[0, max(1.0, float(np.nanmax(kirim_bulanan[kolom_data])) * 1.2)], row=baris, col=1)
    fig.update_xaxes(tickmode="array", tickvals=list(kirim_bulanan.index), ticktext=teks_bulan, tickangle=-45, row=2, col=1)
    tampilkan(gaya(fig, 560))
    st.caption("Bulan mengikuti tanggal pembelian. Kedua sumbu dimulai dari nol.")

# ----------------------------------------------------------------------------
# Tab 3: peta dan jarak (geospatial analysis dan binning)
# ----------------------------------------------------------------------------
with tab_peta:
    st.subheader("Sebaran pesanan dan keterlambatan menurut lokasi")

    metrik_peta = st.radio("Warna bubble", ["Pesanan terlambat (%)", "Rata-rata durasi pengiriman (hari)", "Rata-rata review score"],
                           horizontal=True, key="metrik_peta")
    peta = pesanan.groupby("customer_state").agg(
        pesanan=("order_id", "size"), terlambat=("is_late", "mean"), durasi=("delivery_days", "mean"),
        review=("review_score", "mean"), lat=("customer_lat", "median"), lng=("customer_lng", "median"))
    peta["terlambat"] *= 100
    peta = peta.dropna(subset=["lat", "lng"])
    kolom_warna = {"Pesanan terlambat (%)": "terlambat", "Rata-rata durasi pengiriman (hari)": "durasi",
                   "Rata-rata review score": "review"}[metrik_peta]
    skala_warna = "YlOrRd_r" if kolom_warna == "review" else "YlOrRd"   # warna gelap = kondisi lebih buruk

    cukup = peta[peta["pesanan"] >= MIN_PESANAN_STATE]
    sedikit = peta[peta["pesanan"] < MIN_PESANAN_STATE]
    fig = go.Figure()
    if len(cukup):
        referensi_ukuran = 2.0 * peta["pesanan"].max() / (56.0 ** 2)
        fig.add_trace(go.Scattergeo(
            lat=cukup["lat"], lon=cukup["lng"], mode="markers+text", text=cukup.index, textposition="top center",
            textfont=dict(size=11, color=GELAP),
            marker=dict(size=cukup["pesanan"], sizemode="area", sizeref=referensi_ukuran, sizemin=5,
                        color=cukup[kolom_warna], colorscale=skala_warna, showscale=True,
                        colorbar=dict(title=dict(text=metrik_peta), thickness=14, len=0.7),
                        line=dict(color=GELAP, width=0.8)),
            customdata=np.stack([cukup.index, cukup["pesanan"], cukup["terlambat"], cukup["durasi"], cukup["review"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b><br>Pesanan: %{customdata[1]:,.0f}<br>Terlambat: %{customdata[2]:.1f}%"
                           "<br>Durasi rata-rata: %{customdata[3]:.1f} hari<br>Review rata-rata: %{customdata[4]:.2f}<extra></extra>"),
            name="Negara bagian"))
    if len(sedikit):
        fig.add_trace(go.Scattergeo(
            lat=sedikit["lat"], lon=sedikit["lng"], mode="markers+text", text=sedikit.index, textposition="top center",
            textfont=dict(size=10, color="#718096"),
            marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color="#718096", width=1.2)),
            hovertemplate="<b>%{text}</b><br>Kurang dari 100 pesanan (tidak dibandingkan)<extra></extra>", name="< 100 pesanan"))
    fig.update_geos(scope="south america", lataxis_range=[-35, 7], lonaxis_range=[-75, -32], showcountries=True,
                    countrycolor="#CBD5E0", showland=True, landcolor="#F7FAFC", showocean=True, oceancolor="#EDF2F7",
                    showframe=False)
    tampilkan(gaya(fig, 620))
    st.caption("Luas bubble sebanding dengan jumlah pesanan. Posisi = median koordinat zip prefix pelanggan per negara bagian. "
               f"Lingkaran kosong: kurang dari {MIN_PESANAN_STATE} pesanan.")

    st.markdown("---")
    st.markdown("**Jarak penjual ke pelanggan, durasi pengiriman, dan keterlambatan**")
    pesanan_jarak = pesanan.dropna(subset=["distance_km"]).copy()
    if len(pesanan_jarak) >= 50:
        pesanan_jarak["kelompok_jarak"] = pd.cut(pesanan_jarak["distance_km"], bins=BATAS_JARAK, labels=LABEL_JARAK)
        per_jarak = pesanan_jarak.groupby("kelompok_jarak", observed=False).agg(
            pesanan=("order_id", "size"), durasi=("delivery_days", "mean"), terlambat=("is_late", "mean"),
            review=("review_score", "mean"))
        per_jarak["terlambat"] *= 100
        per_jarak = per_jarak[per_jarak["pesanan"] > 0]
        sumbu_x = [f"{k}<br>n = {angka(n)}" for k, n in zip(per_jarak.index.astype(str), per_jarak["pesanan"])]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Rata-rata durasi pengiriman (hari)", "Pesanan terlambat (%)"))
        fig.add_trace(go.Bar(x=sumbu_x, y=per_jarak["durasi"], marker_color=BIRU, text=[angka(v, 1) for v in per_jarak["durasi"]],
                             textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>%{text} hari<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Bar(x=sumbu_x, y=per_jarak["terlambat"], marker_color=ORANYE, text=[f"{angka(v, 1)}%" for v in per_jarak["terlambat"]],
                             textposition="outside", cliponaxis=False, hovertemplate="%{x}<br>%{text}<extra></extra>"), row=1, col=2)
        fig.update_yaxes(range=[0, float(per_jarak["durasi"].max()) * 1.2], row=1, col=1)
        fig.update_yaxes(range=[0, max(1.0, float(per_jarak["terlambat"].max())) * 1.2], row=1, col=2)
        tampilkan(gaya(fig, 430))
        rho = pesanan_jarak[["distance_km", "delivery_days"]].corr(method="spearman").iloc[0, 1]
        st.caption(f"Korelasi Spearman jarak dan durasi pengiriman: {angka(rho, 2)}. Jarak = garis lurus (haversine) antara zip prefix "
                   "penjual dan pelanggan; pesanan multi-penjual memakai jarak terjauh. Pesanan tanpa data jarak tidak diikutkan.")
    else:
        st.info("Data jarak tidak mencukupi pada kombinasi filter ini.")

    st.markdown("**Pengiriman dalam satu negara bagian dibandingkan lintas negara bagian**")
    rute = pesanan.groupby("is_interstate").agg(
        pesanan=("order_id", "size"), durasi=("delivery_days", "mean"), terlambat=("is_late", "mean"), review=("review_score", "mean"))
    rute.index = rute.index.map({False: "Satu negara bagian", True: "Lintas negara bagian"})
    tabel_rute = pd.DataFrame({
        "Pesanan": rute["pesanan"].map(lambda v: angka(v)),
        "Rata-rata durasi (hari)": rute["durasi"].map(lambda v: angka(v, 1)),
        "Pesanan terlambat": rute["terlambat"].map(lambda v: f"{angka(v * 100, 1)}%"),
        "Rata-rata review": rute["review"].map(lambda v: angka(v, 2) if pd.notna(v) else "-"),
    })
    tabel_rute.index.name = "Rute"
    st.dataframe(tabel_rute)

# ----------------------------------------------------------------------------
# Tab 4: segmentasi pelanggan RFM (Pertanyaan 3)
# ----------------------------------------------------------------------------
with tab_rfm:
    st.subheader("Segmentasi pelanggan dengan RFM")
    st.caption("RFM dihitung pada seluruh periode analisis dengan tanggal acuan 1 September 2018; hanya filter negara bagian yang berlaku. "
               "Batas nilai tinggi (persentil ke-80 pengeluaran) dihitung dari seluruh pelanggan agar definisi segmen tetap sama.")

    rfm = hitung_rfm(dasar, TANGGAL_ACUAN_RFM)
    rfm["segmen"] = segmentasi_rfm(rfm, BATAS_RECENCY_HARI, batas_monetary)

    pelanggan_repeat = rfm["frequency"] >= 2
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pelanggan", angka(len(rfm)))
    k2.metric("Pelanggan membeli ≥2 kali", f"{angka(pelanggan_repeat.mean() * 100, 2)}%")
    k3.metric("Revenue dari pelanggan repeat", f"{angka(rfm.loc[pelanggan_repeat, 'monetary'].sum() / rfm['monetary'].sum() * 100, 1)}%")
    k4.metric("Batas nilai tinggi (P80)", f"R$ {angka(batas_monetary, 2)}")

    ringkasan = rfm.groupby("segmen").agg(
        pelanggan=("frequency", "size"), revenue=("monetary", "sum"), rata_pengeluaran=("monetary", "mean"),
        median_recency=("recency", "median")).reindex(URUTAN_SEGMEN).fillna(0)
    ringkasan["pelanggan_pct"] = ringkasan["pelanggan"] / ringkasan["pelanggan"].sum() * 100
    ringkasan["revenue_pct"] = ringkasan["revenue"] / ringkasan["revenue"].sum() * 100

    kiri, kanan = st.columns([1, 1.7])
    with kiri:
        st.markdown("**Jumlah pesanan per pelanggan**")
        distribusi = rfm["frequency"].clip(upper=3).map({1: "1 pesanan", 2: "2 pesanan", 3: "≥3 pesanan"}).value_counts()
        distribusi = distribusi.reindex(["1 pesanan", "2 pesanan", "≥3 pesanan"]).fillna(0)
        fig = go.Figure(go.Bar(
            x=distribusi.index, y=distribusi.values, marker_color=[BIRU, BIRU_MUDA, BIRU_MUDA],
            text=[f"{angka(v)}<br>({angka(v / distribusi.sum() * 100, 1)}%)" for v in distribusi.values],
            textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:,.0f} pelanggan<extra></extra>"))
        fig.update_yaxes(title="Jumlah pelanggan", range=[0, distribusi.max() * 1.2], tickformat=",.0f")
        tampilkan(gaya(fig, 430))
    with kanan:
        st.markdown("**Porsi pelanggan dan porsi revenue per segmen**")
        urut = ringkasan.sort_values("revenue_pct")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=urut.index, x=urut["pelanggan_pct"], orientation="h", name="Porsi pelanggan", marker_color=ABU,
                             text=[f"{angka(v, 1)}%" for v in urut["pelanggan_pct"]], textposition="outside", cliponaxis=False))
        fig.add_trace(go.Bar(y=urut.index, x=urut["revenue_pct"], orientation="h", name="Porsi revenue", marker_color=BIRU,
                             text=[f"{angka(v, 1)}%" for v in urut["revenue_pct"]], textposition="outside", cliponaxis=False))
        fig.update_xaxes(title="Persen", range=[0, max(urut["pelanggan_pct"].max(), urut["revenue_pct"].max()) * 1.25])
        fig.update_layout(barmode="group", legend=dict(traceorder="reversed"))
        tampilkan(gaya(fig, 430, legenda=True))

    tinggi = ringkasan.loc[["Baru - Nilai Tinggi", "Dormant - Nilai Tinggi"]]
    st.caption(f"Dua segmen bernilai tinggi (Baru dan Dormant) mencakup {angka(tinggi['pelanggan_pct'].sum(), 1)}% pelanggan "
               f"dan menyumbang {angka(tinggi['revenue_pct'].sum(), 1)}% revenue.")

    st.markdown("**Ringkasan segmen**")
    tabel_segmen = pd.DataFrame({
        "Pelanggan": ringkasan["pelanggan"].map(lambda v: angka(v)),
        "% pelanggan": ringkasan["pelanggan_pct"].map(lambda v: angka(v, 1)),
        "Revenue (R$)": ringkasan["revenue"].map(lambda v: angka(v)),
        "% revenue": ringkasan["revenue_pct"].map(lambda v: angka(v, 1)),
        "Rata-rata belanja (R$)": ringkasan["rata_pengeluaran"].map(lambda v: angka(v, 2)),
        "Median recency (hari)": ringkasan["median_recency"].map(lambda v: angka(v)),
    })
    tabel_segmen.index.name = "Segmen"
    st.dataframe(tabel_segmen)

    with st.expander("Aturan segmentasi"):
        st.markdown(
            f"- **Recency**: hari antara 1 September 2018 dan pembelian terakhir. Aktif bila recency ≤ {BATAS_RECENCY_HARI} hari.\n"
            "- **Frequency**: jumlah pesanan valid per pelanggan. Repeat bila ≥ 2.\n"
            f"- **Monetary**: total harga barang pelanggan (R$). Nilai tinggi bila ≥ R$ {angka(batas_monetary, 2)} (persentil ke-80).\n\n"
            "| Segmen | Aturan |\n|---|---|\n"
            "| Loyal - Aktif | frequency ≥ 2 dan aktif |\n"
            "| Loyal - Tidak Aktif | frequency ≥ 2 dan tidak aktif |\n"
            "| Baru - Nilai Tinggi | frequency 1, aktif, nilai tinggi |\n"
            "| Baru - Reguler | frequency 1, aktif, bukan nilai tinggi |\n"
            "| Dormant - Nilai Tinggi | frequency 1, tidak aktif, nilai tinggi |\n"
            "| Dormant - Reguler | frequency 1, tidak aktif, bukan nilai tinggi |")

# ----------------------------------------------------------------------------
# Tab 5: data dan definisi
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Definisi metrik")
    st.markdown(
        "| Istilah | Definisi |\n|---|---|\n"
        "| Pesanan valid | Status `delivered` dan tanggal tiba di pelanggan terisi |\n"
        "| Revenue | Jumlah `price` seluruh item pada pesanan valid, tidak termasuk ongkos kirim (`freight_value`) |\n"
        "| Pesanan terlambat | Tanggal tiba (per hari kalender) lebih besar dari tanggal estimasi pengiriman |\n"
        "| Durasi pengiriman | Selang waktu dari pembelian sampai barang tiba di pelanggan, dalam hari |\n"
        "| Review score | Skor ulasan 1–5; bila satu pesanan memiliki beberapa ulasan, dipakai ulasan terakhir |\n"
        "| Anomali review ID | Sebagian `review_id` terkait ke beberapa `order_id`; ID tidak dihapus secara arbitrer karena tidak ada pemetaan otoritatif. Analisis memilih ulasan terbaru per pesanan. |\n"
        "| Batas interpretasi | Perbedaan review score antara pesanan tepat waktu dan terlambat bersifat deskriptif, bukan bukti kausal. |\n"
        "| Pelanggan | `customer_unique_id` |\n"
        "| Jarak | Garis lurus (haversine) antara zip prefix penjual dan pelanggan, dalam km |")
    st.subheader("Cuplikan data pada filter terpilih")
    st.caption("`main_data.csv` berisi satu baris per item pesanan. Kolom tingkat pesanan (review, durasi, keterlambatan, jarak) "
               "bernilai sama pada seluruh item dalam pesanan yang sama.")
    st.dataframe(data.drop(columns=["tanggal_beli", "bulan"]).head(200))
    st.caption(f"Menampilkan 200 dari {angka(len(data))} baris.")
