"""Dashboard pendamping wawancara pakar (Subbab 3.2.8). Jalankan: streamlit run app.py"""
import json
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
EMAS, ABU, NAIK, TURUN, HIJAU = "#E9B000", "#9B5DE5", "#E63946", "#1D7CF2", "#06A77D"
ORANYE, GARIS, UNGU, TOSKA = "#F77F00", "#3D348B", "#8338EC", "#00A6A6"
FAKTOR_WARNA = {"Gold": "#E9B000", "Silver": "#00A6A6", "Crude_Oil": "#F77F00", "EUR_USD": "#3A86FF", "USD_CNY": "#FF006E",
                "SP500": "#8338EC", "Inflation": "#2DC653", "Iron_Ore": "#B5651D", "GPR": "#C1121F"}
CADANGAN = ["#F15BB5", "#00BBF9", "#9EF01A", "#FB5607", "#7209B7", "#06D6A0"]
SKALA_WARNA = [[0, "#1D7CF2"], [0.5, "#FFD166"], [1, "#E63946"]]
IMP_COL = "Mean|SHAP| (USD)"
LABEL = {
    "Gold": "Harga emas hari ini", "Silver": "Harga perak", "Crude_Oil": "Harga minyak mentah", "EUR_USD": "Nilai tukar EUR/USD",
    "USD_CNY": "Nilai tukar USD/CNY", "SP500": "Indeks S&P 500", "Inflation": "Proksi inflasi (ETF TIPS)",
    "Iron_Ore": "Harga bijih besi", "GPR": "Indeks risiko geopolitik (GPR)",
}
KET = {
    "Gold": "Harga emas pada hari yang sama, sehari sebelum harga yang diperkirakan.", "Silver": "Mewakili kondisi pasar logam mulia.",
    "Crude_Oil": "Berkaitan dengan pasar energi, aktivitas ekonomi, dan tekanan inflasi.", "EUR_USD": "Mewakili pergerakan pasar valuta asing internasional.",
    "USD_CNY": "Nilai tukar Dolar AS terhadap Yuan Tiongkok.", "SP500": "Mewakili kondisi pasar saham Amerika Serikat.",
    "Inflation": "Harga ETF obligasi terindeks inflasi, dipakai sebagai pendekatan inflasi.", "Iron_Ore": "Mewakili kondisi pasar komoditas logam.",
    "GPR": "Mengukur tingkat risiko geopolitik berdasarkan pemberitaan surat kabar.",
}
PAKAR = ["Pakar 1 - praktisi industri emas", "Pakar 2 - praktisi perbankan"]
NILAI_SESUAI = ["Sesuai", "Sebagian sesuai", "Tidak sesuai"]
ARAH_PAKAR = ["Searah", "Berlawanan", "Tidak menentu", "Tidak ada hubungan"]

KOLOM_SHEET = ["responden", "tanggal", "waktu_simpan", "gold_ditampilkan", "faktor", "peringkat_model", "pengaruh_usd", "porsi_persen",
               "arah_hasil", "kesesuaian", "arah_pakar", "peringkat_pakar", "catatan"]


def simpan_ke_sheets(hasil):
    """Menambahkan baris penilaian ke Google Sheets. Mengembalikan (berhasil, pesan)."""
    try:
        akun = dict(st.secrets["gcp_service_account"])
        url = st.secrets["sheet"]["url"]
    except Exception:
        return False, "belum diatur"
    try:
        import gspread
        ws = gspread.service_account_from_dict(akun).open_by_url(url).sheet1
        if ws.row_values(1) != KOLOM_SHEET:
            if any(ws.row_values(1)):
                return False, "baris pertama Google Sheets tidak sesuai, gunakan sheet kosong"
            ws.update([KOLOM_SHEET], "A1")
        isi = hasil.reindex(columns=KOLOM_SHEET).astype(object).where(hasil.reindex(columns=KOLOM_SHEET).notna(), "")
        ws.append_rows([[int(v) if isinstance(v, float) and v.is_integer() else v if isinstance(v, (int, float, str)) else str(v) for v in baris] for baris in isi.values.tolist()], value_input_option="RAW")
        return True, "Tersimpan ke Google Sheets"
    except Exception as e:
        return False, f"gagal menyimpan ke Google Sheets ({type(e).__name__})"


st.set_page_config(page_title="Faktor yang Berkaitan dengan Harga Emas", layout="wide")


def lb(n):
    return LABEL.get(n, n)


def warna(n):
    return FAKTOR_WARNA.get(n, CADANGAN[sum(map(ord, n)) % len(CADANGAN)])


def wajib(nama):
    path = BASE / nama
    if not path.exists():
        st.error(f"File {nama} tidak ditemukan. Simpan file itu di folder yang sama dengan app.py.")
        st.stop()
    return path


@st.cache_data
def muat_data():
    shap_df = pd.read_csv(wajib("xai_shap_values.csv"), parse_dates=["Date"])
    imp = pd.read_csv(wajib("xai_feature_importance.csv"))
    test = pd.read_csv(wajib("test_scaled.csv"), parse_dates=["Date"])
    fitur = imp["Fitur"].tolist()
    kurang = [c for c in ["actual", "pred", "base"] + [f"shap_{n}" for n in fitur] if c not in shap_df] + [n for n in fitur if n not in test]
    if kurang:
        raise ValueError(f"Kolom tidak ditemukan: {kurang}")
    if len(shap_df) != len(test) or not shap_df["Date"].equals(test["Date"]):
        raise ValueError("Tanggal xai_shap_values.csv dan test_scaled.csv tidak sama. Pastikan keduanya dari split yang sama.")
    nilai, asli = test[fitur].copy(), False
    path = BASE / "scaler_X.pkl"
    if path.exists():
        try:
            sx = joblib.load(path)
            nama = list(getattr(sx, "feature_names_in_", test.columns.drop(["Date", "Target"], errors="ignore")))
            if sx.n_features_in_ == len(fitur) and set(nama) == set(fitur):
                nilai = pd.DataFrame(sx.inverse_transform(test[nama].to_numpy()), columns=nama)[fitur]
                asli = True
        except Exception:
            pass
    if (BASE / "xai_korelasi_fitur.csv").exists():
        korelasi, sumber_kor = pd.read_csv(BASE / "xai_korelasi_fitur.csv", index_col=0).loc[fitur, fitur], "data latih"
    else:
        korelasi, sumber_kor = test[fitur].corr(), "data uji"
    if (BASE / "train_scaled.csv").exists():
        latih = pd.read_csv(BASE / "train_scaled.csv")
        ke_emas, sumber_emas = latih[fitur].corrwith(latih["Target"]), "data latih"
    elif "Target" in test:
        ke_emas, sumber_emas = test[fitur].corrwith(test["Target"]), "data uji"
    else:
        ke_emas, sumber_emas = korelasi["Gold"], "data latih" if sumber_kor == "data latih" else "data uji"
    harian = None
    if (BASE / "prediksi_harian.csv").exists():
        harian = pd.read_csv(BASE / "prediksi_harian.csv", parse_dates=["Date", "date_next"])
        hilang = [c for c in ["Date", "gold_today", "date_next", "pred_next", "actual_next"] if c not in harian]
        if hilang:
            raise ValueError(f"Kolom prediksi_harian.csv tidak ditemukan: {hilang}")
        if len(harian) != len(shap_df) or not harian["Date"].equals(shap_df["Date"]):
            raise ValueError("Tanggal prediksi_harian.csv dan xai_shap_values.csv tidak sama. Pastikan keduanya dari model dan data uji yang sama.")
    return shap_df, imp, test, nilai, fitur, asli, korelasi, sumber_kor, ke_emas, sumber_emas, harian


def tampilkan(fig, key, tinggi=440, kanan=10, legenda=False):
    fig.update_layout(height=tinggi, margin=dict(l=10, r=kanan, t=30, b=10), hoverlabel=dict(align="left"), showlegend=legenda,
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, key=key)


CLS_URUT = ["0.9", "0.6", "0.3"]


@st.cache_data
def harga_emas_harian(harian, test):
    """Harga emas sebenarnya (USD) untuk SEMUA hari perdagangan, dari kolom Gold di train/val/test_scaled.
    Skala 0-1 dikembalikan ke USD dengan hubungan linear yang dicocokkan pada 200 hari uji (gold_today)."""
    gs = test["Gold"].to_numpy(float); gu = harian["gold_today"].to_numpy(float)
    a, b = np.polyfit(gs, gu, 1)
    if np.abs(a * gs + b - gu).max() > 0.01:
        raise ValueError("Kolom Gold di test_scaled.csv tidak cocok dengan gold_today di prediksi_harian.csv.")
    bagian, sumber = [test[["Date", "Gold"]]], ["data uji"]
    for nama, label in [("train_scaled.csv", "data latih"), ("val_scaled.csv", "data validasi")]:
        if (BASE / nama).exists():
            bagian.append(pd.read_csv(BASE / nama, usecols=["Date", "Gold"], parse_dates=["Date"])); sumber.append(label)
    semua = pd.concat(bagian).drop_duplicates("Date").sort_values("Date")
    seri = pd.Series(a * semua["Gold"].to_numpy(float) + b, index=pd.DatetimeIndex(semua["Date"]))
    besok = harian.dropna(subset=["date_next"])                                 # harga hari berikutnya dari hari uji
    seri = pd.concat([seri, pd.Series(besok["actual_next"].to_numpy(float), index=pd.DatetimeIndex(besok["date_next"]))])
    seri = seri[~seri.index.duplicated(keep="first")].sort_index()
    lengkap = len(sumber) == 3
    return seri, ("gabungan " + ", ".join(sumber)) if lengkap else ("hanya " + ", ".join(sumber) + "; tambahkan train_scaled.csv dan val_scaled.csv agar semua hari tampil")


try:
    df, imp, test, nilai, fitur, asli, korelasi, sumber_kor, ke_emas, sumber_emas, harian = muat_data()
except ValueError as e:
    st.error(str(e))
    st.stop()

phi = df[[f"shap_{n}" for n in fitur]].to_numpy()
y, p = df["actual"].to_numpy(), df["pred"].to_numpy()
tanggal = df["Date"].dt.strftime("%Y-%m-%d").tolist()
err = y - p
mae, mape = np.abs(err).mean(), np.mean(np.abs(err / y)) * 100
skala = "nilai asli" if asli else "skala 0 sampai 1 (0 terendah, 1 tertinggi pada data latih)"


def rho_arah(n):
    return pd.Series(nilai[n].to_numpy()).rank().corr(pd.Series(phi[:, fitur.index(n)]).rank())


def arah(n):
    r = rho_arah(n)
    if r >= 0.3:
        return f"Saat {lb(n)} lebih tinggi, perkiraan harga emas cenderung lebih tinggi."
    if r <= -0.3:
        return f"Saat {lb(n)} lebih tinggi, perkiraan harga emas cenderung lebih rendah."
    return f"Arah pengaruh {lb(n)} tidak konsisten pada data ini."


def arah_hasil(n):
    r = rho_arah(n)
    if r >= 0.3:
        return "Searah"
    if r <= -0.3:
        return "Berlawanan"
    return "Tidak konsisten"


def kekuatan(r):
    a = abs(r)
    kuat = "kuat" if a >= 0.7 else "sedang" if a >= 0.4 else "lemah"
    return f"{'searah' if r > 0 else 'berlawanan arah'}, {kuat}" if a >= 0.1 else "hampir tidak berkaitan"


def peran(q):
    return "besar" if q >= 25 else "sedang" if q >= 10 else "kecil"


st.sidebar.title("Faktor harga emas")
halaman = st.sidebar.radio("Tahapan", ["Panduan", "Prediksi harian", "Peringkat faktor", "Keterkaitan faktor", "Bandingkan faktor", "Contoh per tanggal", "Penilaian pakar"])
ada_gold = "Gold" in fitur
sertakan_gold = st.sidebar.checkbox("Tampilkan harga emas hari ini", value=False) if ada_gold else False
st.sidebar.caption("Hasil ini berasal dari analisis data historis 2019-2026. Hasil menunjukkan keterkaitan, bukan bukti sebab-akibat, bukan ramalan harga masa depan, dan bukan rekomendasi investasi.")

data = (imp if sertakan_gold else imp[imp["Fitur"] != "Gold"]).reset_index(drop=True)
tampil = data["Fitur"].tolist()
porsi = (data[IMP_COL] / data[IMP_COL].sum() * 100).to_numpy()
pengaruh = data[IMP_COL].to_numpy()
CATATAN_GOLD = "Harga emas hari ini tetap ikut diperhitungkan, tetapi tidak ditampilkan di sini agar perbedaan antarfaktor lain lebih mudah dinilai."

if halaman == "Panduan":
    st.title("Faktor apa yang paling berkaitan dengan perkiraan harga emas?")
    st.write("Penelitian ini menganalisis data harian 2019-2026 untuk memperkirakan harga emas satu hari ke depan dari sembilan faktor. Bapak/Ibu tidak perlu memahami cara perhitungannya. Yang kami minta adalah penilaian: apakah faktor-faktor yang muncul dalam hasil ini sesuai dengan pengetahuan dan pengalaman Bapak/Ibu tentang pergerakan harga emas.")
    st.subheader("Cara membaca angkanya")
    contoh = tampil[0]
    st.info(f"Contoh: {lb(contoh)} bernilai {pengaruh[0]:.2f} USD. Artinya, faktor ini rata-rata menggeser perkiraan harga emas sekitar {pengaruh[0]:.2f} USD per hari, bisa naik atau turun. Semakin besar angkanya, semakin besar perannya dalam perkiraan harga emas.")
    st.subheader("Tahapan sesi")
    st.markdown("1. **Prediksi harian:** melihat harga emas hari ini, perkiraan harga besok, dan faktor yang paling berpengaruh setiap hari.\n2. **Peringkat faktor:** melihat urutan faktor dari yang paling besar pengaruhnya.\n3. **Keterkaitan faktor:** melihat faktor yang bergerak bersama harga emas dan satu sama lain.\n4. **Bandingkan faktor:** melihat perbedaan dua faktor.\n5. **Contoh per tanggal:** melihat faktor yang berperan pada hari tertentu.\n6. **Penilaian pakar:** menilai kesesuaian hasil dengan pengetahuan Bapak/Ibu tentang pasar emas.")
    st.subheader("Yang perlu diingat")
    st.markdown(f"- Rata-rata perkiraan harga emas meleset sekitar **{mae:.2f} USD ({mape:.2f}%)** dari harga aktual pada data uji.\n- Peringkat menunjukkan faktor yang paling berkaitan dengan perkiraan harga emas, **bukan bukti bahwa faktor tersebut menyebabkan** harga emas bergerak.\n- Faktor yang bergerak bersama bisa berbagi pengaruh, sehingga pengaruh masing-masing bisa terlihat lebih kecil atau lebih besar.\n- Tidak ada jawaban benar atau salah. Pendapat yang berbeda dari model juga merupakan hasil penting.")
    if not sertakan_gold and ada_gold:
        st.caption(CATATAN_GOLD)

elif halaman == "Prediksi harian":
    st.title("Harga emas hari ini dan perkiraan harga besok")
    if harian is None:
        st.warning("File prediksi_harian.csv belum ada. Jalankan notebook buat_prediksi_harian.ipynb, lalu simpan prediksi_harian.csv di folder yang sama dengan app.py.")
        st.stop()
    st.write(f"Halaman ini menampilkan {len(harian)} hari pada data uji. Untuk setiap hari, harga emas hari itu dipakai untuk memperkirakan harga emas pada hari perdagangan berikutnya, lalu dibandingkan dengan harga yang sebenarnya terjadi.")
    st.caption("Ini adalah pengujian pada data historis yang sudah diketahui jawabannya, bukan ramalan harga di masa depan. Data uji dipilih acak dari 2019-2026, jadi tanggalnya tidak selalu berurutan.")

    hj, hp, ha = harian["gold_today"].to_numpy(), harian["pred_next"].to_numpy(), harian["actual_next"].to_numpy()
    ada_int = "lo_0.9" in harian and "hi_0.9" in harian
    lo9, hi9 = (harian["lo_0.9"].to_numpy(), harian["hi_0.9"].to_numpy()) if ada_int else (None, None)
    tgl_besok = harian["date_next"].dt.strftime("%Y-%m-%d").fillna("hari berikutnya").tolist()
    e_h = ha - hp
    gap_xai = np.abs(hp - p)
    if gap_xai.max() > 0.5:
        k_model = None
        try:
            k_model = json.loads((BASE / "xai_config.json").read_text()).get("k_models")
        except Exception:
            pass
        if k_model:
            st.info(f"Angka perkiraan berasal dari rata-rata 1.000 model. Faktor pengaruh dihitung dari rata-rata {k_model} model pertama, "
                    f"yang perkiraannya rata-rata berbeda {gap_xai.mean():,.2f} USD (paling jauh {gap_xai.max():,.2f} USD).")
        else:
            st.warning(f"Perkiraan di prediksi_harian.csv berbeda dengan perkiraan pada hasil analisis faktor (maks {gap_xai.max():.2f} USD). Pastikan keduanya berasal dari model yang sama.")
    arah_ok = np.mean(np.sign(hp - hj) == np.sign(ha - hj)) * 100
    k = st.columns(4)
    k[0].metric("Rata-rata meleset", f"{np.abs(e_h).mean():,.2f} USD")
    k[1].metric("Rata-rata meleset (%)", f"{np.mean(np.abs(e_h / ha)) * 100:.2f}%")
    k[2].metric("Arah naik/turun tepat", f"{arah_ok:.1f}%")
    if ada_int:
        k[3].metric("Harga sebenarnya di dalam rentang 90%", f"{np.mean((ha >= lo9) & (ha <= hi9)) * 100:.1f}%")

    idx_f = [fitur.index(n) for n in tampil]
    sub = phi[:, idx_f]
    top = np.abs(sub).argmax(1)
    top_kode = [tampil[j] for j in top]

    st.subheader("Pilih hari")
    pintas = {"Pilih tanggal": None, "Perkiraan paling meleset": int(np.abs(e_h).argmax()), "Harga emas tertinggi": int(hj.argmax()), "Harga emas terendah": int(hj.argmin())}
    cara = st.radio("Tampilkan", list(pintas), horizontal=True, key="pintas_harian")
    i = tanggal.index(st.selectbox("Tanggal (hari ini)", tanggal, key="tgl_harian")) if pintas[cara] is None else pintas[cara]

    mode_grafik = st.radio("Grafik", ["1 hari", "Rentang 1 minggu"], horizontal=True, key="mode_grafik_harian")
    hari_ini, besok = harian["Date"].iloc[i], harian["date_next"].iloc[i]
    if pd.isna(besok):
        besok = hari_ini + pd.Timedelta(days=1)

    if mode_grafik == "1 hari":
        fig = go.Figure()
        if ada_int:
            for cl, alpha in [("0.9", 0.16), ("0.6", 0.24), ("0.3", 0.34)]:
                if f"lo_{cl}" in harian and f"hi_{cl}" in harian:
                    lo_c, hi_c = float(harian[f"lo_{cl}"].iloc[i]), float(harian[f"hi_{cl}"].iloc[i])
                    fig.add_trace(go.Scatter(x=[besok, besok], y=[lo_c, hi_c], mode="lines", line=dict(color=f"rgba(131,56,236,{alpha + 0.3})", width=28 - 8 * CLS_URUT.index(cl)),
                                             name=f"Rentang perkiraan {int(float(cl) * 100)}%",
                                             hovertemplate=f"Rentang {int(float(cl) * 100)}%: {lo_c:,.2f} - {hi_c:,.2f} USD<extra></extra>"))
        fig.add_trace(go.Scatter(x=[hari_ini, besok], y=[hj[i], hp[i]], mode="lines+markers+text", name="Perkiraan harga besok",
                                 line=dict(color=UNGU, width=3, dash="dash"), marker=dict(size=[13, 15], color=[EMAS, UNGU]),
                                 text=["", f"{hp[i]:,.2f}"], textposition="middle left", textfont=dict(color=UNGU, size=13),
                                 hovertemplate="Perkiraan: %{y:,.2f} USD<extra></extra>"))
        fig.add_trace(go.Scatter(x=[hari_ini, besok], y=[hj[i], ha[i]], mode="lines+markers+text", name="Harga sebenarnya besok",
                                 line=dict(color=TOSKA, width=3), marker=dict(size=[13, 15], color=[EMAS, TOSKA], symbol=["circle", "diamond"]),
                                 text=[f"{hj[i]:,.2f}", f"{ha[i]:,.2f}"], textposition=["middle left", "middle right"], textfont=dict(color=[EMAS, TOSKA], size=13),
                                 hovertemplate="Harga: %{y:,.2f} USD<extra></extra>"))
        fig.update_xaxes(title_text="", tickvals=[hari_ini, besok], ticktext=[f"Hari ini<br>{hari_ini:%d %b %Y}", f"Besok<br>{besok:%d %b %Y}"],
                         range=[hari_ini - (besok - hari_ini) * 0.6, besok + (besok - hari_ini) * 0.6])
        fig.update_yaxes(title_text="Harga emas (USD)")
        fig.update_layout(hovermode="closest")
        tampilkan(fig, f"harian_1hari_{i}", 460, legenda=True)
        st.caption("Titik kuning adalah harga emas hari ini. Garis ungu putus-putus menuju perkiraan harga besok, garis toska menuju harga yang sebenarnya terjadi. Pita ungu adalah rentang perkiraan (makin gelap makin sempit).")
    else:
        awal, akhir = hari_ini - pd.Timedelta(days=3), max(hari_ini + pd.Timedelta(days=3), besok)
        seri, sumber_seri = harga_emas_harian(harian, test)
        minggu = seri[(seri.index >= awal) & (seri.index <= akhir)]
        dlm = harian[(harian["Date"] >= awal) & (harian["Date"] <= akhir)].index.to_numpy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=minggu.index, y=minggu.values, mode="lines+markers", name="Harga emas sebenarnya (harian)",
                                 line=dict(color=EMAS, width=2.5), marker=dict(size=8),
                                 hovertemplate="%{x|%a, %d %b %Y}<br>Harga emas: %{y:,.2f} USD<extra></extra>"))
        if ada_int:
            for j in dlm:
                bsk = harian["date_next"].iloc[j] if pd.notna(harian["date_next"].iloc[j]) else harian["Date"].iloc[j] + pd.Timedelta(days=1)
                fig.add_trace(go.Scatter(x=[bsk, bsk], y=[lo9[j], hi9[j]], mode="lines", line=dict(color="rgba(131,56,236,0.35)", width=18),
                                         name="Rentang perkiraan 90%", showlegend=bool(j == dlm[0]),
                                         hovertemplate=f"Rentang 90%: {lo9[j]:,.2f} - {hi9[j]:,.2f} USD<extra></extra>"))
        for j in dlm:
            bsk = harian["date_next"].iloc[j] if pd.notna(harian["date_next"].iloc[j]) else harian["Date"].iloc[j] + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(x=[harian["Date"].iloc[j], bsk], y=[hj[j], hp[j]], mode="lines+markers", name="Perkiraan harga besok",
                                     showlegend=bool(j == dlm[0]), line=dict(color=UNGU, width=3 if j == i else 2, dash="dash"),
                                     marker=dict(size=[0, 14 if j == i else 10], color=UNGU, symbol="star"),
                                     customdata=[[tanggal[j], lb(top_kode[j])]] * 2,
                                     hovertemplate="Perkiraan dari %{customdata[0]}: %{y:,.2f} USD<br>Faktor paling berpengaruh: %{customdata[1]}<extra></extra>"))
        fig.add_vrect(x0=hari_ini - pd.Timedelta(hours=12), x1=hari_ini + pd.Timedelta(hours=12), fillcolor=NAIK, opacity=0.10, line_width=0)
        fig.update_xaxes(title_text="", type="date", dtick=86400000, tickformat="%a<br>%d %b", range=[awal - pd.Timedelta(hours=12), akhir + pd.Timedelta(hours=12)])
        fig.update_yaxes(title_text="Harga emas (USD)")
        fig.update_layout(hovermode="closest")
        tampilkan(fig, f"harian_minggu_{i}", 460, legenda=True)
        lain = len(dlm) - 1
        st.caption(f"Rentang {awal:%d %b} - {akhir:%d %b %Y}. Garis kuning adalah harga emas sebenarnya setiap hari perdagangan ({sumber_seri}). "
                   f"Bintang ungu adalah perkiraan model untuk hari berikutnya; kolom merah muda menandai hari yang dipilih."
                   + (f" Ada {lain} hari uji lain di minggu ini yang juga ditampilkan perkiraannya." if lain > 0 else " Hari lain di minggu ini bukan bagian data uji, jadi tidak punya perkiraan."))

    st.subheader(f"Hari ini: {tanggal[i]}  →  perkiraan untuk {tgl_besok[i]}")
    k = st.columns(4)
    k[0].metric("Harga emas hari ini", f"{hj[i]:,.2f} USD")
    k[1].metric("Perkiraan harga besok", f"{hp[i]:,.2f} USD", f"{hp[i] - hj[i]:+,.2f} USD dari hari ini")
    k[2].metric("Harga sebenarnya besok", f"{ha[i]:,.2f} USD", f"{ha[i] - hj[i]:+,.2f} USD dari hari ini")
    k[3].metric("Selisih perkiraan", f"{hp[i] - ha[i]:+,.2f} USD", delta_color="off")
    if ada_int:
        st.write(f"Rentang perkiraan 90%: **{lo9[i]:,.2f} - {hi9[i]:,.2f} USD**. Harga sebenarnya {'berada di dalam' if lo9[i] <= ha[i] <= hi9[i] else 'berada di luar'} rentang ini.")

    s_i = sub[i]; o = np.argsort(np.abs(s_i))[::-1]
    v, kode = s_i[o], [tampil[j] for j in o]
    naik = [lb(kode[j]) for j in range(len(v)) if v[j] > 0][:2]
    turun = [lb(kode[j]) for j in range(len(v)) if v[j] < 0][:2]
    st.write(f"Faktor paling berpengaruh pada hari ini adalah **{lb(kode[0])}** ({v[0]:+,.2f} USD). Pendorong perkiraan lebih tinggi: **{', '.join(naik) or 'tidak ada'}**. Pendorong perkiraan lebih rendah: **{', '.join(turun) or 'tidak ada'}**.")
    isi = [float(nilai[n].iloc[i]) for n in kode]
    fig = go.Figure(go.Bar(
        x=v, y=[lb(n) for n in kode], orientation="h", marker_color=[warna(n) for n in kode],
        text=[f"{a:+,.2f} USD" for a in v], textposition="outside", cliponaxis=False, customdata=isi,
        hovertemplate="<b>%{y}</b><br>Dorongan ke perkiraan harga emas: %{x:+,.2f} USD<br>Nilai faktor: %{customdata:,.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=GARIS, line_width=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Dorongan ke perkiraan harga emas besok (USD)")
    tampilkan(fig, f"faktor_harian_{i}_{sertakan_gold}", 420, kanan=90)
    keterangan = "Batang ke kanan mendorong perkiraan lebih tinggi dari rata-ratanya, batang ke kiri mendorong lebih rendah. Setiap faktor punya warnanya sendiri."
    if ada_gold and not sertakan_gold:
        keterangan += f" Pengaruh harga emas hari ini ({float(phi[i, fitur.index('Gold')]):+,.2f} USD) tidak ditampilkan."
    st.caption(keterangan)

    st.subheader("Perubahan pengaruh faktor dibanding hari lain")
    urut = np.argsort(harian["Date"].to_numpy())
    pos = int(np.flatnonzero(urut == i)[0])
    pilihan_banding = ["Hari uji sebelumnya", "Pilih tanggal lain"]
    cara_b = st.radio("Bandingkan dengan", pilihan_banding, horizontal=True, key="cara_banding")
    if cara_b == "Hari uji sebelumnya":
        if pos == 0:
            st.info("Hari ini adalah hari uji paling awal, jadi tidak ada hari uji sebelumnya. Pilih tanggal lain untuk membandingkan.")
            j = None
        else:
            j = int(urut[pos - 1])
    else:
        lain = [t for t in tanggal if t != tanggal[i]]
        j = tanggal.index(st.selectbox("Tanggal pembanding", lain, index=max(0, min(len(lain) - 1, pos - 1)), key="tgl_banding"))
    if j is not None:
        jarak = (harian["Date"].iloc[i] - harian["Date"].iloc[j]).days
        s_j = sub[j]; d = s_i - s_j
        o2 = np.argsort(np.abs(d))[::-1]; kode2 = [tampil[q] for q in o2]
        k = st.columns(3)
        k[0].metric("Tanggal pembanding", tanggal[j], f"{abs(jarak)} hari {'sebelumnya' if jarak > 0 else 'sesudahnya'}", delta_color="off")
        k[1].metric("Harga emas hari itu", f"{hj[j]:,.2f} USD", f"{hj[i] - hj[j]:+,.2f} USD ke hari ini", delta_color="off")
        k[2].metric("Perkiraan besok pada hari itu", f"{hp[j]:,.2f} USD", f"{hp[i] - hp[j]:+,.2f} USD ke hari ini", delta_color="off")
        fig = go.Figure(go.Bar(
            x=d[o2], y=[lb(n) for n in kode2], orientation="h", marker_color=[warna(n) for n in kode2],
            text=[f"{v:+,.2f} USD" for v in d[o2]], textposition="outside", cliponaxis=False,
            customdata=np.column_stack([s_j[o2], s_i[o2]]),
            hovertemplate="<b>%{y}</b><br>" + f"{tanggal[j]}" + ": %{customdata[0]:+,.2f} USD<br>" + f"{tanggal[i]}" + ": %{customdata[1]:+,.2f} USD<br>Perubahan: %{x:+,.2f} USD<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=GARIS, line_width=1)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text=f"Perubahan dorongan dari {tanggal[j]} ke {tanggal[i]} (USD)")
        tampilkan(fig, f"banding_{i}_{j}_{sertakan_gold}", 400, kanan=90)
        naik_b = [(lb(tampil[q]), d[q]) for q in o2 if d[q] > 0][:2]
        turun_b = [(lb(tampil[q]), d[q]) for q in o2 if d[q] < 0][:2]
        teks = lambda L: ", ".join(f"**{n}** ({v:+,.2f} USD)" for n, v in L) or "tidak ada"
        st.write(f"Dibanding {tanggal[j]}, faktor yang pengaruhnya paling bertambah ke arah perkiraan lebih tinggi: {teks(naik_b)}. "
                 f"Yang paling bergeser ke arah perkiraan lebih rendah: {teks(turun_b)}. "
                 f"Faktor paling berpengaruh: **{lb(top_kode[j])}** pada {tanggal[j]} dan **{lb(top_kode[i])}** pada {tanggal[i]}.")
        ket_b = ("Batang ke kanan berarti faktor itu mendorong perkiraan lebih tinggi dibanding hari pembanding, batang ke kiri berarti mendorong lebih rendah. "
                 "Arahkan kursor ke batang untuk melihat besar dorongan pada kedua tanggal.")
        if jarak > 7:
            ket_b += f" Data uji dipilih acak, jadi hari uji sebelumnya berjarak {jarak} hari; perubahan bisa juga mencerminkan perubahan kondisi pasar selama rentang itu."
        st.caption(ket_b)

    st.subheader("Faktor paling berpengaruh di setiap hari")
    st.write("Setiap titik adalah satu hari. Warna dan baris menunjukkan faktor yang paling berpengaruh pada hari itu, dan ukuran titik menunjukkan besar pengaruhnya.")
    besar = np.abs(sub[np.arange(len(sub)), top])
    ukuran = 6 + 14 * (besar - besar.min()) / ((besar.max() - besar.min()) or 1)
    fig = go.Figure()
    for n in tampil:
        msk = np.array(top_kode) == n
        if not msk.any():
            continue
        fig.add_trace(go.Scatter(
            x=harian["Date"][msk], y=[lb(n)] * int(msk.sum()), mode="markers", name=lb(n), marker=dict(color=warna(n), size=ukuran[msk], opacity=0.85),
            customdata=np.column_stack([np.array(tanggal)[msk], sub[msk, tampil.index(n)]]),
            hovertemplate="<b>%{customdata[0]}</b><br>" + lb(n) + ": %{customdata[1]:+,.2f} USD<extra></extra>",
        ))
    fig.update_xaxes(title_text="Tanggal (hari ini)", type="date")
    tampilkan(fig, f"top_harian_{sertakan_gold}", 360)

    hit = pd.Series(top_kode).value_counts().reindex(tampil).fillna(0).astype(int)
    hit = hit[hit > 0].sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=hit.values, y=[lb(n) for n in hit.index], orientation="h", marker_color=[warna(n) for n in hit.index],
                           text=[f"{c} hari ({c / len(sub) * 100:.0f}%)" for c in hit.values], textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>Paling berpengaruh pada %{x} hari<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Jumlah hari sebagai faktor paling berpengaruh")
    tampilkan(fig, f"hitung_top_{sertakan_gold}", 340, kanan=110)
    if ada_gold and not sertakan_gold:
        st.caption(CATATAN_GOLD)

    st.subheader(f"Tabel prediksi {len(harian)} hari")
    tabel = pd.DataFrame({
        "Tanggal (hari ini)": tanggal, "Harga emas hari ini (USD)": np.round(hj, 2), "Tanggal besok": tgl_besok,
        "Perkiraan besok (USD)": np.round(hp, 2),
        **({"Batas bawah 90% (USD)": np.round(lo9, 2), "Batas atas 90% (USD)": np.round(hi9, 2)} if ada_int else {}),
        "Harga sebenarnya besok (USD)": np.round(ha, 2), "Selisih (USD)": np.round(hp - ha, 2),
        "Arah perkiraan": np.where(hp >= hj, "naik", "turun"), "Arah sebenarnya": np.where(ha >= hj, "naik", "turun"),
        "Faktor paling berpengaruh": [lb(n) for n in top_kode], "Pengaruhnya (USD)": np.round(sub[np.arange(len(sub)), top], 2),
    })
    st.dataframe(tabel, hide_index=True)
    st.download_button("Unduh tabel prediksi (CSV)", tabel.to_csv(index=False).encode("utf-8-sig"), "prediksi_harian_200_hari.csv", "text/csv")

elif halaman == "Peringkat faktor":
    st.title("Peringkat faktor yang berkaitan dengan harga emas")
    st.write(f"Berdasarkan hasil analisis, faktor dengan pengaruh terbesar terhadap perkiraan harga emas{'' if sertakan_gold else ' selain harga emas hari ini'} adalah **{lb(tampil[0])}**, diikuti **{lb(tampil[1])}** dan **{lb(tampil[2])}**.")
    nama = [lb(n) for n in tampil]
    fig = go.Figure(go.Bar(
        x=pengaruh, y=nama, orientation="h", marker_color=[warna(n) for n in tampil],
        text=[f"{v:.2f} USD ({q:.1f}%)" for v, q in zip(pengaruh, porsi)], textposition="outside", cliponaxis=False,
        customdata=[[arah(n), KET.get(n, "")] for n in tampil],
        hovertemplate="<b>%{y}</b><br>Rata-rata pengaruh: %{x:.2f} USD<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Rata-rata pergeseran perkiraan harga emas per hari (USD)")
    tampilkan(fig, "peringkat", kanan=130)
    st.caption("Persentase adalah porsi terhadap faktor yang ditampilkan. Arahkan kursor ke batang untuk melihat penjelasan.")
    st.subheader("Sebaran pengaruh setiap hari")
    st.write("Setiap titik adalah satu hari pada data uji. Titik di kanan garis tengah berarti faktor tersebut mendorong perkiraan harga emas lebih tinggi dari rata-ratanya, sedangkan titik di kiri berarti mendorong lebih rendah. Warna menunjukkan nilai faktor pada hari itu: merah berarti tinggi, biru berarti rendah.")
    st.caption("Contoh cara membaca: kalau titik merah banyak berada di kanan, berarti saat nilai faktor itu tinggi, perkiraan harga emas cenderung lebih tinggi. Semakin lebar sebaran titiknya, semakin besar perannya.")
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for r_, n in enumerate(tampil):
        j = fitur.index(n)
        s_ = phi[:, j]
        hist, tepi = np.histogram(s_, bins=40)
        rapat = hist[np.clip(np.digitize(s_, tepi) - 1, 0, len(hist) - 1)] / max(hist.max(), 1)
        fig.add_trace(go.Scatter(
            x=s_, y=r_ + rng.uniform(-0.38, 0.38, len(s_)) * rapat, mode="markers", name=lb(n),
            marker=dict(size=7, color=np.clip(test[n], 0, 1), colorscale=SKALA_WARNA, cmin=0, cmax=1, showscale=r_ == 0,
                        colorbar=dict(title="Nilai faktor", tickvals=[0, 0.5, 1], ticktext=["Rendah", "Sedang", "Tinggi"])),
            customdata=[[lb(n), tanggal[i], float(nilai[n].iloc[i]), y[i], p[i]] for i in range(len(df))],
            hovertemplate="<b>%{customdata[0]}</b><br>Tanggal: %{customdata[1]}<br>Dorongan ke perkiraan harga emas: %{x:+.2f} USD<br>Nilai faktor: %{customdata[2]:,.4f}<br>Harga emas sebenarnya: %{customdata[3]:,.2f} USD<br>Perkiraan harga emas: %{customdata[4]:,.2f} USD<extra></extra>",
        ))
    fig.add_vline(x=0, line_color=GARIS, line_width=1)
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(tampil))), ticktext=nama, autorange="reversed")
    fig.update_xaxes(title_text="Dorongan ke perkiraan harga emas (USD)")
    tampilkan(fig, f"beeswarm_{sertakan_gold}", 520)

    st.subheader("Arah pengaruh tiap faktor")
    st.dataframe(pd.DataFrame({"Peringkat": range(1, len(tampil) + 1), "Faktor": nama, "Pengaruh (USD)": np.round(pengaruh, 2), "Porsi (%)": np.round(porsi, 1), "Arah keterkaitan dengan harga emas": [arah(n) for n in tampil]}), hide_index=True)
    if not sertakan_gold and ada_gold:
        st.caption(CATATAN_GOLD)

elif halaman == "Keterkaitan faktor":
    st.title("Faktor yang bergerak bersama")
    st.write("Halaman ini melihat seberapa kuat dua hal cenderung naik atau turun bersamaan, diukur dengan korelasi dari -1 sampai 1. Nilai mendekati 1 berarti searah, mendekati -1 berarti berlawanan arah, dan mendekati 0 berarti hampir tidak berkaitan.")
    st.info("Korelasi melihat satu faktor saja. Peringkat faktor melihat tambahan informasi suatu faktor setelah faktor lain ikut diperhitungkan. Jadi faktor yang sangat berkaitan dengan harga emas bisa punya peran kecil kalau informasinya sudah terwakili faktor lain. Keduanya juga bukan bukti sebab-akibat.")

    st.subheader("Keterkaitan langsung dengan harga emas esok hari")
    peringkat_model = {n: i + 1 for i, n in enumerate(tampil)}
    urut = sorted(tampil, key=lambda n: -abs(ke_emas[n]))
    r_emas = np.array([ke_emas[n] for n in urut])
    fig = go.Figure(go.Bar(
        x=r_emas, y=[lb(n) for n in urut], orientation="h", marker_color=[NAIK if v > 0 else TURUN for v in r_emas],
        text=[f"{v:+.2f}" for v in r_emas], textposition="outside", cliponaxis=False,
        customdata=[[kekuatan(ke_emas[n]), peringkat_model[n]] for n in urut],
        hovertemplate="<b>%{y}</b><br>Korelasi: %{x:+.2f} (%{customdata[0]})<br>Peringkat pengaruh: %{customdata[1]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=GARIS, line_width=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Korelasi dengan harga emas esok hari", range=[-1.1, 1.1])
    tampilkan(fig, f"kor_emas_{sertakan_gold}", kanan=40)
    st.caption(f"Dihitung dari {sumber_emas}. Merah berarti searah, biru berarti berlawanan arah. Kekuatan: kuat jika |korelasi| ≥ 0,7, sedang jika 0,4 sampai 0,7, lemah jika di bawah 0,4.")

    rank_kor = {n: i + 1 for i, n in enumerate(urut)}
    tabel = pd.DataFrame({"Faktor": [lb(n) for n in tampil], "Korelasi dengan harga emas": [round(float(ke_emas[n]), 2) for n in tampil],
                          "Peringkat menurut korelasi": [rank_kor[n] for n in tampil], "Peringkat pengaruh": [peringkat_model[n] for n in tampil]})
    st.dataframe(tabel, hide_index=True)
    beda = [n for n in tampil if abs(rank_kor[n] - peringkat_model[n]) >= 3]
    if beda:
        st.write("Faktor yang peringkat pengaruhnya cukup berbeda dari keterkaitan langsungnya: " + ", ".join(
            f"**{lb(n)}** (peringkat korelasi {rank_kor[n]}, peringkat pengaruh {peringkat_model[n]})" for n in beda) + ".")

    st.subheader("Keterkaitan antarfaktor")
    nama = [lb(n) for n in tampil]
    z = korelasi.loc[tampil, tampil].to_numpy()
    teks = [[kekuatan(z[i, j]) if i != j else "faktor yang sama" for j in range(len(tampil))] for i in range(len(tampil))]
    fig = go.Figure(go.Heatmap(
        z=z, x=nama, y=nama, zmin=-1, zmax=1, zmid=0, colorscale=SKALA_WARNA, texttemplate="%{z:.2f}", customdata=teks,
        colorbar=dict(title="Korelasi"), hovertemplate="<b>%{y}</b> dan <b>%{x}</b><br>Korelasi: %{z:.2f}<br>%{customdata}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    tampilkan(fig, f"heatmap_{sertakan_gold}", 560)
    st.caption(f"Dihitung dari {sumber_kor}. Arahkan kursor ke kotak untuk melihat penjelasan.")

    catatan = []
    for i, a in enumerate(tampil):
        for b in tampil[i + 1:]:
            r = float(korelasi.loc[a, b])
            if abs(r) >= 0.7:
                catatan.append(f"**{lb(a)}** dan **{lb(b)}** bergerak {'searah' if r > 0 else 'berlawanan arah'} kuat (korelasi {r:.2f}).")
    if ada_gold and not sertakan_gold:
        for n in tampil:
            r = float(korelasi.loc["Gold", n])
            if abs(r) >= 0.7:
                catatan.append(f"**{lb(n)}** sangat berkaitan dengan harga emas hari ini (korelasi {r:.2f}), yang tetap ikut diperhitungkan meskipun tidak ditampilkan.")
    if catatan:
        st.subheader("Yang perlu diperhatikan")
        st.markdown("\n".join(f"- {c}" for c in catatan) + "\n\nFaktor yang bergerak bersama bisa berbagi pengaruh, sehingga peringkat di antara faktor-faktor tersebut perlu dibaca bersama-sama.")

elif halaman == "Bandingkan faktor":
    st.title("Bandingkan dua faktor")
    kiri, kanan = st.columns(2)
    a = kiri.selectbox("Faktor pertama", tampil, index=0, format_func=lb)
    b = kanan.selectbox("Faktor kedua", tampil, index=1, format_func=lb)
    if a == b:
        st.warning("Pilih dua faktor yang berbeda.")
        st.stop()
    va, vb = float(pengaruh[tampil.index(a)]), float(pengaruh[tampil.index(b)])
    besar, kecil, vb_, vk_ = (a, b, va, vb) if va >= vb else (b, a, vb, va)
    rasio = f"sekitar {vb_ / vk_:.1f} kali" if vk_ > 0 else "jauh lebih besar dari"
    st.write(f"Pengaruh **{lb(besar)}** ({vb_:.2f} USD) {rasio} pengaruh **{lb(kecil)}** ({vk_:.2f} USD).")
    r = float(korelasi.loc[a, b])
    if abs(r) >= 0.7:
        st.info(f"Kedua faktor ini cenderung bergerak {'searah' if r > 0 else 'berlawanan'} pada {sumber_kor} (korelasi {r:.2f}), sehingga pengaruh keduanya bisa saling berbagi. Selisih peringkatnya perlu dibaca hati-hati.")
    else:
        st.caption(f"Keterkaitan kedua faktor pada {sumber_kor}: korelasi {r:.2f} ({kekuatan(r)}).")
    sa, sb = phi[:, fitur.index(a)], phi[:, fitur.index(b)]
    a_naik_b_turun = int(((sa > 0) & (sb < 0)).sum())
    a_turun_b_naik = int(((sa < 0) & (sb > 0)).sum())
    n_hari = len(sa)
    n_lawan = a_naik_b_turun + a_turun_b_naik
    k = st.columns(3)
    k[0].metric("Hari dengan arah berlawanan", f"{n_lawan} dari {n_hari} hari", f"{n_lawan / n_hari * 100:.1f}%", delta_color="off")
    k[1].metric(f"{lb(a)} menaikkan, {lb(b)} menurunkan", f"{a_naik_b_turun} hari")
    k[2].metric(f"{lb(a)} menurunkan, {lb(b)} menaikkan", f"{a_turun_b_naik} hari")

    st.subheader("Pengaruh kedua faktor per tanggal")
    status = ["Arah berlawanan" if u * w < 0 else "Arah sama" for u, w in zip(sa, sb)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tanggal, y=sa, mode="lines+markers", name=lb(a), line=dict(color=warna(a), width=1.5), marker=dict(size=5), customdata=status,
                             hovertemplate=lb(a) + ": %{y:+.2f} USD<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tanggal, y=sb, mode="lines+markers", name=lb(b), line=dict(color=warna(b), width=1.5), marker=dict(size=5),
                             hovertemplate=lb(b) + ": %{y:+.2f} USD<extra></extra>"))
    fig.add_hline(y=0, line_color=GARIS, line_width=1)
    fig.update_xaxes(title_text="Tanggal pada data uji", type="date")
    fig.update_yaxes(title_text="Dorongan ke perkiraan harga emas (USD)")
    fig.update_layout(hovermode="x unified")
    tampilkan(fig, f"gabung_{a}_{b}", 460, legenda=True)
    st.caption("Di atas garis nol berarti faktor mendorong perkiraan harga emas lebih tinggi dari rata-ratanya. Di bawah garis nol berarti mendorong lebih rendah. Ini bukan berarti harga emas naik atau turun dibanding hari sebelumnya. Data uji dipilih acak, jadi tanggalnya tidak selalu berurutan.")

    with st.expander("Lihat hubungan nilai tiap faktor dengan pengaruhnya"):
        st.caption(f"Tiap titik adalah satu hari. Nilai faktor memakai {skala}.")
        lim = [min(sa.min(), sb.min()), max(sa.max(), sb.max())]
        pad = (lim[1] - lim[0]) * 0.05 or 1
        kol_a, kol_b = st.columns(2)
        for kol, n in ((kol_a, a), (kol_b, b)):
            with kol:
                st.subheader(lb(n))
                st.write(arah(n))
                fig = go.Figure(go.Scatter(
                    x=nilai[n], y=phi[:, fitur.index(n)], mode="markers", marker=dict(size=7, opacity=0.8, color=warna(n)),
                    customdata=[[tanggal[i], y[i], p[i]] for i in range(len(df))],
                    hovertemplate="<b>%{customdata[0]}</b><br>Nilai faktor: %{x:,.4f}<br>Dorongan ke perkiraan harga emas: %{y:+.2f} USD<br>Harga emas sebenarnya: %{customdata[1]:,.2f} USD<br>Perkiraan harga emas: %{customdata[2]:,.2f} USD<extra></extra>",
                ))
                fig.add_hline(y=0, line_color=GARIS, line_width=1)
                fig.update_xaxes(title_text="Nilai faktor")
                fig.update_yaxes(title_text="Dorongan ke perkiraan harga emas (USD)", range=[lim[0] - pad, lim[1] + pad])
                tampilkan(fig, f"banding_{n}", 400)

elif halaman == "Contoh per tanggal":
    st.title("Faktor yang berperan pada hari tertentu")
    pintas = {"Pilih tanggal": None, "Perkiraan paling meleset": int(np.abs(err).argmax()), "Perkiraan dengan selisih menengah": int(np.argsort(np.abs(err))[len(err) // 2])}
    if "GPR" in fitur:
        pintas["Risiko geopolitik tertinggi"] = int(test["GPR"].to_numpy().argmax())
    cara = st.radio("Contoh", list(pintas), horizontal=True)
    i = tanggal.index(st.selectbox("Tanggal", tanggal)) if pintas[cara] is None else pintas[cara]
    dasar = float(df["base"].iloc[i])
    gold_i = float(phi[i, fitur.index("Gold")]) if ada_gold and not sertakan_gold else 0.0
    awal = dasar + gold_i
    st.subheader(tanggal[i])
    k = st.columns(3)
    k[0].metric("Harga emas sebenarnya esok hari", f"{y[i]:,.2f} USD")
    k[1].metric("Perkiraan harga emas", f"{p[i]:,.2f} USD")
    k[2].metric("Selisih", f"{p[i] - y[i]:+,.2f} USD")

    s = phi[i, [fitur.index(n) for n in tampil]]
    o = np.argsort(np.abs(s))[::-1]
    v, kode = s[o], [tampil[j] for j in o]
    naik = [lb(kode[j]) for j in range(len(v)) if v[j] > 0][:2]
    turun = [lb(kode[j]) for j in range(len(v)) if v[j] < 0][:2]
    st.write(f"Pada hari ini, faktor utama yang mendorong perkiraan harga emas lebih tinggi: **{', '.join(naik) or 'tidak ada'}**. Faktor utama yang mendorong perkiraan harga emas lebih rendah: **{', '.join(turun) or 'tidak ada'}**.")
    kiri_batang = awal + np.r_[0, np.cumsum(v)[:-1]]
    isi = [float(nilai[n].iloc[i]) for n in kode]
    fig = go.Figure(go.Bar(
        x=v, y=[lb(n) for n in kode], base=kiri_batang, orientation="h", marker_color=[NAIK if a > 0 else TURUN for a in v],
        customdata=np.column_stack([isi, kiri_batang, kiri_batang + v]),
        hovertemplate="<b>%{y}</b><br>Pengaruh: %{x:+.2f} USD<br>Nilai faktor: %{customdata[0]:,.4f}<br>Perkiraan harga emas bergeser dari %{customdata[1]:,.2f} ke %{customdata[2]:,.2f} USD<extra></extra>",
    ))
    fig.add_vline(x=awal, line_dash="dash", line_color=ABU, annotation_text="Titik awal", annotation_position="top left")
    fig.add_vline(x=p[i], line_color=GARIS, annotation_text="Perkiraan", annotation_position="top right")
    fig.add_vline(x=y[i], line_color=HIJAU, annotation_text="Sebenarnya", annotation_position="bottom right")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Harga emas (USD)")
    tampilkan(fig, f"lokal_{i}_{sertakan_gold}")
    if ada_gold and not sertakan_gold:
        st.caption(f"Titik awal {awal:,.2f} USD adalah rata-rata perkiraan harga emas {dasar:,.2f} USD ditambah pengaruh harga emas hari ini {gold_i:+,.2f} USD. Merah menaikkan perkiraan, biru menurunkannya. Nilai faktor memakai {skala}.")
    else:
        st.caption(f"Titik awal adalah rata-rata perkiraan harga emas. Merah menaikkan perkiraan, biru menurunkannya. Nilai faktor memakai {skala}.")

else:
    st.title("Penilaian pakar")
    st.write("Bagian ini tidak menilai ketepatan perkiraan harga. Kami ingin mengetahui apakah hubungan tiap faktor dengan harga emas dalam hasil ini sesuai dengan pengetahuan dan pengalaman Bapak/Ibu.")
    st.caption("Tidak ada jawaban benar atau salah. Pendapat yang berbeda dari hasil ini juga sangat berguna.")
    n_f = len(tampil)
    with st.form("penilaian"):
        k1, k2 = st.columns(2)
        pakar = k1.selectbox("Responden", PAKAR)
        tgl = k2.date_input("Tanggal wawancara", value=date.today())
        jawaban = []
        for r, n in enumerate(tampil):
            st.markdown(f"**{r + 1}. {lb(n)}**")
            ah = arah_hasil(n)
            teks_arah = {"Searah": "searah dengan harga emas", "Berlawanan": "berlawanan arah dengan harga emas", "Tidak konsisten": "tanpa arah yang konsisten"}[ah]
            st.write(f"Dalam hasil ini, {lb(n)} memiliki peran {peran(porsi[r])} (peringkat {r + 1} dari {n_f}) dan berkaitan {teks_arah}.")
            a1, a2, a3, a4 = st.columns([3, 3, 1, 3])
            jawaban.append((n, r + 1, ah,
                            a1.radio("Apakah sesuai dengan pengetahuan Bapak/Ibu?", NILAI_SESUAI, index=None, key=f"s_{n}"),
                            a2.radio("Menurut Bapak/Ibu, hubungannya dengan harga emas", ARAH_PAKAR, index=None, key=f"a_{n}"),
                            a3.selectbox("Peringkat menurut Bapak/Ibu", ["-"] + list(range(1, n_f + 1)), key=f"r_{n}"),
                            a4.text_input("Alasan atau contoh dari pengalaman", key=f"c_{n}")))
        st.subheader("Penilaian keseluruhan")
        top3 = st.radio(f"Apakah {', '.join(lb(n) for n in tampil[:3])} memang termasuk faktor utama yang berkaitan dengan harga emas menurut pengetahuan Bapak/Ibu?", NILAI_SESUAI, index=None)
        urutan = st.radio("Apakah urutan keseluruhan faktor sesuai dengan pengetahuan dan pengalaman Bapak/Ibu?", NILAI_SESUAI, index=None)
        beda = st.text_area("Faktor mana yang menurut Bapak/Ibu seharusnya berperan lebih besar atau lebih kecil terhadap harga emas, dan mengapa?")
        tambah = st.text_area("Adakah faktor lain yang menurut Bapak/Ibu penting bagi harga emas tetapi belum termasuk di sini?")
        kirim = st.form_submit_button("Simpan penilaian")
    if kirim:
        kosong = [lb(j[0]) for j in jawaban if j[3] is None or j[4] is None]
        if kosong or top3 is None or urutan is None:
            st.warning("Penilaian belum disimpan. Mohon lengkapi pilihan kesesuaian dan hubungan untuk semua faktor, serta dua pertanyaan keseluruhan.")
        else:
            ringkas = {"responden": pakar, "tanggal": str(tgl), "waktu_simpan": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "gold_ditampilkan": sertakan_gold}
            baris = [{**ringkas, "faktor": lb(n), "peringkat_model": rk, "pengaruh_usd": round(float(pengaruh[rk - 1]), 2), "porsi_persen": round(float(porsi[rk - 1]), 1), "arah_hasil": ah_, "kesesuaian": s_, "arah_pakar": ap, "peringkat_pakar": "" if rp == "-" else rp, "catatan": c_} for n, rk, ah_, s_, ap, rp, c_ in jawaban]
            for t, nilai_j in (("Tiga faktor teratas", top3), ("Urutan keseluruhan", urutan), ("Faktor yang perlu naik/turun", beda), ("Faktor tambahan", tambah)):
                baris.append({**ringkas, "faktor": f"(umum) {t}", "kesesuaian": nilai_j})
            hasil = pd.DataFrame(baris)
            rp_all = [j[5] for j in jawaban]
            if "-" not in rp_all and len(set(rp_all)) == n_f:
                rs = pd.Series([j[1] for j in jawaban], dtype=float).corr(pd.Series(rp_all, dtype=float))
                st.session_state["kesesuaian"] = float(rs)
            else:
                st.session_state["kesesuaian"] = None
            nama_file = f"penilaian_{pakar[:7].replace(' ', '_').lower()}_{datetime.now():%Y%m%d_%H%M%S}.csv"
            st.session_state["hasil_csv"] = hasil.to_csv(index=False).encode("utf-8-sig")
            st.session_state["nama_file"] = nama_file
            ok, pesan = simpan_ke_sheets(hasil)
            lokal = True
            try:
                folder = BASE / "hasil_penilaian"
                folder.mkdir(exist_ok=True)
                (folder / nama_file).write_bytes(st.session_state["hasil_csv"])
            except OSError:
                lokal = False
            if ok:
                st.success("Penilaian tersimpan ke Google Sheets. Terima kasih.")
            elif pesan == "belum diatur" and lokal:
                st.info(f"Penilaian tersimpan di folder hasil_penilaian sebagai {nama_file}. Google Sheets belum diatur, jadi kalau dashboard dibuka online, mohon unduh file di bawah ini.")
            else:
                st.warning(f"Penilaian belum tersimpan otomatis: {pesan}. Mohon unduh file di bawah ini.")
    if "hasil_csv" in st.session_state:
        st.download_button("Unduh hasil penilaian (CSV)", st.session_state["hasil_csv"], st.session_state["nama_file"], "text/csv")
        if st.session_state.get("kesesuaian") is not None:
            st.caption(f"Untuk peneliti: kesesuaian peringkat model dan pakar (korelasi Spearman) = {st.session_state['kesesuaian']:.3f}.")