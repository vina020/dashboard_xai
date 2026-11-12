"""Dashboard pendamping wawancara pakar (Subbab 3.2.8). Jalankan: streamlit run app.py"""
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
EMAS, ABU, NAIK, TURUN, HIJAU = "#B8860B", "#7A8594", "#C0392B", "#2C6FAB", "#2E8B57"
ORANYE = "#D9822B"
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

st.set_page_config(page_title="Faktor yang Berkaitan dengan Harga Emas", layout="wide")


def lb(n):
    return LABEL.get(n, n)


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
    return shap_df, imp, test, nilai, fitur, asli, korelasi, sumber_kor, ke_emas, sumber_emas


def tampilkan(fig, key, tinggi=440, kanan=10, legenda=False):
    fig.update_layout(height=tinggi, margin=dict(l=10, r=kanan, t=30, b=10), hoverlabel=dict(align="left"), showlegend=legenda,
                      legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(fig, key=key)


try:
    df, imp, test, nilai, fitur, asli, korelasi, sumber_kor, ke_emas, sumber_emas = muat_data()
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
halaman = st.sidebar.radio("Tahapan", ["Panduan", "Peringkat faktor", "Keterkaitan faktor", "Bandingkan faktor", "Contoh per tanggal", "Penilaian pakar"])
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
    st.write("Penelitian ini menganalisis data harian 2019-2026 untuk memperkirakan harga emas satu hari ke depan dari sembilan faktor. Hal yang perlu diperhatikan oleh Bapak/Ibu adalah: apakah faktor-faktor yang muncul dalam hasil ini sesuai dengan pengetahuan dan pengalaman Bapak/Ibu tentang pergerakan harga emas.")
    st.subheader("Cara membaca angkanya")
    contoh = tampil[0]
    st.info(f"Contoh: {lb(contoh)} bernilai {pengaruh[0]:.2f} USD. Artinya, faktor ini rata-rata menggeser perkiraan harga emas sekitar {pengaruh[0]:.2f} USD per hari, bisa naik atau turun. Semakin besar angkanya, semakin besar perannya dalam perkiraan harga emas.")
    st.subheader("Tahapan sesi")
    st.markdown("1. **Peringkat faktor:** melihat urutan faktor dari yang paling besar pengaruhnya.\n2. **Keterkaitan faktor:** melihat faktor yang bergerak bersama harga emas dan satu sama lain.\n3. **Bandingkan faktor:** melihat perbedaan dua faktor.\n4. **Contoh per tanggal:** melihat faktor yang berperan pada hari tertentu.\n5. **Penilaian pakar:** menilai kesesuaian hasil dengan pengetahuan Bapak/Ibu tentang pasar emas.")
    st.subheader("Yang perlu diingat")
    st.markdown(f"- Rata-rata perkiraan harga emas meleset sekitar **{mae:.2f} USD ({mape:.2f}%)** dari harga aktual pada data uji.\n- Peringkat menunjukkan faktor yang paling berkaitan dengan perkiraan harga emas, **bukan bukti bahwa faktor tersebut menyebabkan** harga emas bergerak.\n- Faktor yang bergerak bersama bisa berbagi pengaruh, sehingga pengaruh masing-masing bisa terlihat lebih kecil atau lebih besar.\n- Tidak ada jawaban benar atau salah. Pendapat yang berbeda dari model juga merupakan hasil penting.")
    if not sertakan_gold and ada_gold:
        st.caption(CATATAN_GOLD)

elif halaman == "Peringkat faktor":
    st.title("Peringkat faktor yang berkaitan dengan harga emas")
    st.write(f"Berdasarkan hasil analisis, faktor dengan pengaruh terbesar terhadap perkiraan harga emas{'' if sertakan_gold else ' selain harga emas hari ini'} adalah **{lb(tampil[0])}**, diikuti **{lb(tampil[1])}** dan **{lb(tampil[2])}**.")
    nama = [lb(n) for n in tampil]
    fig = go.Figure(go.Bar(
        x=pengaruh, y=nama, orientation="h", marker_color=[EMAS if n == "Gold" else ABU for n in tampil],
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
            marker=dict(size=7, color=np.clip(test[n], 0, 1), colorscale="RdBu_r", cmin=0, cmax=1, showscale=r_ == 0,
                        colorbar=dict(title="Nilai faktor", tickvals=[0, 0.5, 1], ticktext=["Rendah", "Sedang", "Tinggi"])),
            customdata=[[lb(n), tanggal[i], float(nilai[n].iloc[i]), y[i], p[i]] for i in range(len(df))],
            hovertemplate="<b>%{customdata[0]}</b><br>Tanggal: %{customdata[1]}<br>Dorongan ke perkiraan harga emas: %{x:+.2f} USD<br>Nilai faktor: %{customdata[2]:,.4f}<br>Harga emas sebenarnya: %{customdata[3]:,.2f} USD<br>Perkiraan harga emas: %{customdata[4]:,.2f} USD<extra></extra>",
        ))
    fig.add_vline(x=0, line_color="black", line_width=1)
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
    fig.add_vline(x=0, line_color="black", line_width=1)
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
        z=z, x=nama, y=nama, zmin=-1, zmax=1, zmid=0, colorscale="RdBu_r", texttemplate="%{z:.2f}", customdata=teks,
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
    fig.add_trace(go.Scatter(x=tanggal, y=sa, mode="lines+markers", name=lb(a), line=dict(color=TURUN, width=1.5), marker=dict(size=5), customdata=status,
                             hovertemplate=lb(a) + ": %{y:+.2f} USD<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tanggal, y=sb, mode="lines+markers", name=lb(b), line=dict(color=ORANYE, width=1.5), marker=dict(size=5),
                             hovertemplate=lb(b) + ": %{y:+.2f} USD<extra></extra>"))
    fig.add_hline(y=0, line_color="black", line_width=1)
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
                    x=nilai[n], y=phi[:, fitur.index(n)], mode="markers", marker=dict(size=7, opacity=0.75, color=EMAS if n == "Gold" else TURUN),
                    customdata=[[tanggal[i], y[i], p[i]] for i in range(len(df))],
                    hovertemplate="<b>%{customdata[0]}</b><br>Nilai faktor: %{x:,.4f}<br>Dorongan ke perkiraan harga emas: %{y:+.2f} USD<br>Harga emas sebenarnya: %{customdata[1]:,.2f} USD<br>Perkiraan harga emas: %{customdata[2]:,.2f} USD<extra></extra>",
                ))
                fig.add_hline(y=0, line_color="black", line_width=1)
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
    fig.add_vline(x=p[i], line_color="black", annotation_text="Perkiraan", annotation_position="top right")
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
            ringkas = {"responden": pakar, "tanggal": str(tgl), "gold_ditampilkan": sertakan_gold}
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
            try:
                folder = BASE / "hasil_penilaian"
                folder.mkdir(exist_ok=True)
                (folder / nama_file).write_bytes(st.session_state["hasil_csv"])
                st.success(f"Penilaian tersimpan di folder hasil_penilaian sebagai {nama_file}. Terima kasih.")
            except OSError:
                st.warning("Penilaian belum tersimpan di komputer. Unduh file di bawah ini.")
    if "hasil_csv" in st.session_state:
        st.download_button("Unduh hasil penilaian (CSV)", st.session_state["hasil_csv"], st.session_state["nama_file"], "text/csv")
        if st.session_state.get("kesesuaian") is not None:
            st.caption(f"Untuk peneliti: kesesuaian peringkat model dan pakar (korelasi Spearman) = {st.session_state['kesesuaian']:.3f}.")