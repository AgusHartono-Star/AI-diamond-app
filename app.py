import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════
st.set_page_config(
    page_title="Diamond Price AI",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════
# LOAD CSS
# ══════════════════════════════════════
css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ══════════════════════════════════════
# CONSTANTS — harus sama dengan train_model.py
# ══════════════════════════════════════
CUT_ORDER     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
COLOR_ORDER   = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
CLARITY_ORDER = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

# ══════════════════════════════════════
# LOAD RESOURCES (cache)
# ══════════════════════════════════════
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_diamonds_final.pkl")
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diamonds.csv")
    df = pd.read_csv(data_path)
    return df.drop_duplicates()

@st.cache_data
def load_results():
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_results.csv")
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return None

# ── Load ──
model = load_model()
df    = load_data()
results_df = load_results()

# ══════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════
st.markdown("""
<div class="hero-title">💎 Diamond Price AI</div>
<div class="hero-subtitle">
    Prediksi harga berlian secara instan menggunakan Machine Learning (XGBoost)
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════
# STATS ROW
# ══════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💎 Total Data", f"{len(df):,}", "baris berlian")
with col2:
    st.metric("📈 Model", "XGBoost", "Akurasi tertinggi")
with col3:
    st.metric("🏷️ Harga Min", f"${df['price'].min():,}")
with col4:
    st.metric("💰 Harga Maks", f"${df['price'].max():,}")

st.markdown("---")

# ══════════════════════════════════════
# TABS
# ══════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediksi Harga",
    "📊 Eksplorasi Data",
    "🏆 Evaluasi Model",
    "ℹ️ Tentang"
])

# ─────────────────────────────────────
# TAB 1 — PREDIKSI
# ─────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">🔮 Prediksi Harga Berlian</div>', unsafe_allow_html=True)
    st.markdown("Masukkan karakteristik berlian, lalu klik **Prediksi Harga**.")
    st.markdown("")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("🔧 Karakteristik Berlian")

        carat = st.slider("⚖️ Carat (Berat)", min_value=0.20, max_value=5.00,
                          value=1.00, step=0.01,
                          help="Berat berlian. Semakin berat, umumnya semakin mahal.")
        cut   = st.selectbox("✂️ Cut (Potongan)", CUT_ORDER, index=4,
                             help="Kualitas potongan. Ideal = terbaik.")
        color = st.selectbox("🎨 Color (Warna)", COLOR_ORDER, index=6,
                             help="Warna berlian. D = tidak berwarna (terbaik), J = sedikit berwarna.")
        clarity = st.selectbox("🔬 Clarity (Kejernihan)", CLARITY_ORDER, index=7,
                               help="IF = paling jernih, I1 = paling banyak inklusi.")
        depth = st.slider("📐 Depth (%)", min_value=40.0, max_value=80.0,
                          value=61.5, step=0.1)
        table = st.slider("📏 Table (%)", min_value=40.0, max_value=100.0,
                          value=57.0, step=0.5)

        st.markdown("**📦 Dimensi (mm)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            x = st.number_input("Length (x)", min_value=0.0, max_value=15.0,
                                value=6.5, step=0.01)
        with c2:
            y = st.number_input("Width (y)", min_value=0.0, max_value=15.0,
                                value=6.5, step=0.01)
        with c3:
            z = st.number_input("Height (z)", min_value=0.0, max_value=10.0,
                                value=4.0, step=0.01)

    with col_right:
        st.subheader("💰 Hasil Prediksi")

        input_df = pd.DataFrame([{
            "carat": carat, "cut": cut, "color": color, "clarity": clarity,
            "depth": depth, "table": table, "x": x, "y": y, "z": z
        }])

        # Preview input
        with st.expander("🔍 Lihat Input Data"):
            st.dataframe(input_df, use_container_width=True)

        if st.button("💎 Prediksi Harga Sekarang", use_container_width=True, type="primary"):
            try:
                price = model.predict(input_df)[0]
                price = max(0, price)          # harga tidak boleh negatif

                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Estimasi Harga Berlian</div>
                    <div class="value">${price:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # Rentang harga (±10%)
                low  = price * 0.90
                high = price * 1.10
                st.info(f"📉 Rentang estimasi: **${low:,.0f}** — **${high:,.0f}** (±10%)")

                # Perbandingan dengan rata-rata dataset
                avg_price = df['price'].mean()
                if price > avg_price:
                    diff_pct = ((price - avg_price) / avg_price) * 100
                    st.success(f"⬆️ {diff_pct:.1f}% **di atas** rata-rata harga dataset (${avg_price:,.0f})")
                else:
                    diff_pct = ((avg_price - price) / avg_price) * 100
                    st.warning(f"⬇️ {diff_pct:.1f}% **di bawah** rata-rata harga dataset (${avg_price:,.0f})")

            except (ValueError, TypeError) as e:
                st.error(f"❌ Prediksi gagal: {e}")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")

        st.markdown("")
        st.markdown("---")

        # Feature Importance
        st.subheader("📊 Fitur Paling Berpengaruh")
        try:
            reg = model.named_steps["reg"]
            importances = reg.feature_importances_
            feature_names = ['carat', 'depth', 'table', 'x', 'y', 'z',
                             'cut', 'color', 'clarity']
            imp_df = pd.DataFrame({'Fitur': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values('Importance', ascending=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#6366f1' if v == imp_df['Importance'].max() else '#a5b4fc'
                      for v in imp_df['Importance']]
            ax.barh(imp_df['Fitur'], imp_df['Importance'], color=colors)
            ax.set_xlabel("Importance Score")
            ax.set_title("Feature Importance — XGBoost")
            ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        except (AttributeError, KeyError):
            st.info("Feature importance tersedia untuk model XGBoost / Random Forest.")

# ─────────────────────────────────────
# TAB 2 — EKSPLORASI DATA
# ─────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">📊 Eksplorasi Dataset Diamonds</div>',
                unsafe_allow_html=True)

    # Info ringkas
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Jumlah Baris", f"{len(df):,}")
    with col_b:
        st.metric("Jumlah Kolom", df.shape[1])
    with col_c:
        st.metric("Missing Value", df.isnull().sum().sum())

    st.markdown("#### 🔍 Sample Data (10 Baris Pertama)")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### 📈 Statistik Deskriptif")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("")
    st.markdown("#### 📉 Visualisasi")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("**Distribusi Harga Berlian**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['price'], bins=60, kde=True, color='#6366f1', ax=ax1)
        ax1.set_xlabel("Price (USD)")
        ax1.set_ylabel("Frekuensi")
        ax1.set_title("Distribusi Harga Berlian")
        ax1.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1)

    with viz_col2:
        st.markdown("**Rata-rata Harga per Cut**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        cut_price = df.groupby('cut')['price'].mean().reindex(CUT_ORDER)
        bars = ax2.bar(cut_price.index, cut_price.values,
                       color=['#c7d2fe', '#a5b4fc', '#818cf8', '#6366f1', '#4f46e5'])
        ax2.set_xlabel("Cut")
        ax2.set_ylabel("Rata-rata Harga (USD)")
        ax2.set_title("Harga Rata-rata per Kualitas Potongan")
        ax2.bar_label(bars, fmt='${:,.0f}', padding=3, fontsize=8)
        ax2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)

    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        st.markdown("**Carat vs Harga**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sample = df.sample(min(2000, len(df)), random_state=42)
        ax3.scatter(sample['carat'], sample['price'],
                    alpha=0.3, s=10, color='#6366f1')
        ax3.set_xlabel("Carat")
        ax3.set_ylabel("Price (USD)")
        ax3.set_title("Hubungan Carat dan Harga")
        ax3.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)

    with viz_col4:
        st.markdown("**Heatmap Korelasi**")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        corr = df.select_dtypes(include=np.number).corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title("Korelasi Fitur Numerik")
        plt.tight_layout()
        st.pyplot(fig4)

# ─────────────────────────────────────
# TAB 3 — EVALUASI MODEL
# ─────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🏆 Evaluasi & Perbandingan Model</div>',
                unsafe_allow_html=True)

    if results_df is not None:
        st.markdown("Hasil pelatihan **3 model** (KNN, Random Forest, XGBoost) "
                    "pada **4 skenario split** dataset.")
        st.markdown("")

        # Tabel lengkap
        st.markdown("#### 📋 Tabel Hasil Evaluasi")
        styled = results_df.sort_values('R2 Score', ascending=False).reset_index(drop=True)
        st.dataframe(styled, use_container_width=True)

        st.markdown("")

        # Bar chart R2
        st.markdown("#### 📊 Perbandingan R² Score per Model & Skenario")
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        palette = {'KNN': '#a5b4fc', 'Random Forest': '#818cf8', 'XGBoost': '#4f46e5'}
        for model_name in results_df['Model'].unique():
            subset = results_df[results_df['Model'] == model_name]
            ax5.plot(subset['Skenario (Train:Test)'], subset['R2 Score'],
                     marker='o', label=model_name,
                     color=palette.get(model_name, '#6366f1'), linewidth=2)
        ax5.set_ylim(0.80, 1.0)
        ax5.set_xlabel("Skenario (Train:Test)")
        ax5.set_ylabel("R² Score")
        ax5.set_title("Perbandingan Akurasi Model di Berbagai Skenario Split")
        ax5.legend()
        ax5.spines[['top', 'right']].set_visible(False)
        ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5)

        # Best model summary
        best_row = results_df.loc[results_df['R2 Score'].idxmax()]
        st.success(f"🏆 **Model Terbaik: {best_row['Model']}** — "
                   f"Skenario {best_row['Skenario (Train:Test)']} — "
                   f"R² = {best_row['R2 Score']:.4f}, "
                   f"RMSE = {best_row['RMSE']:,.0f}, "
                   f"MAE = {best_row['MAE']:,.0f}")
    else:
        st.warning("⚠️ File `model_results.csv` tidak ditemukan. "
                   "Jalankan dulu `python train_model.py` untuk menghasilkan data evaluasi.")
        st.code("python train_model.py", language="bash")

# ─────────────────────────────────────
# TAB 4 — TENTANG
# ─────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">ℹ️ Tentang Proyek</div>', unsafe_allow_html=True)

    about_col1, about_col2 = st.columns(2)

    with about_col1:
        st.markdown("""
        ### 🎓 Deskripsi
        Web aplikasi ini memprediksi **harga berlian** berdasarkan karakteristik fisiknya
        menggunakan model **XGBoost Regressor** yang dilatih pada dataset *Diamonds* dari ggplot2.

        ### 📦 Dataset
        - **Sumber:** ggplot2 Diamonds Dataset
        - **Jumlah baris:** 53,940+ baris
        - **Fitur:** 9 fitur (numerik & kategorikal)
        - **Target:** Harga berlian (USD)

        ### 🤖 Model yang Diuji
        | Model | Deskripsi |
        |-------|-----------|
        | K-Nearest Neighbors | Instance-based learning |
        | Random Forest | Ensemble bagging |
        | **XGBoost ✅** | Gradient boosting — **akurasi tertinggi** |

        ### 📏 Metrik Evaluasi
        - **R² Score** — Koefisien determinasi
        - **RMSE** — Root Mean Squared Error
        - **MAE** — Mean Absolute Error
        """)

    with about_col2:
        st.markdown("""
        ### 💎 Fitur Input Berlian

        | Fitur | Tipe | Keterangan |
        |-------|------|------------|
        | Carat | Float | Berat berlian |
        | Cut | Kategori | Kualitas potongan |
        | Color | Kategori | Warna (J–D) |
        | Clarity | Kategori | Kejernihan |
        | Depth | Float | Total kedalaman (%) |
        | Table | Float | Lebar meja atas (%) |
        | x, y, z | Float | Dimensi (mm) |

        ### 🔧 Tech Stack
        """)
        for badge in ["Python 3.10+", "Streamlit", "XGBoost", "Scikit-learn", "Pandas", "Matplotlib"]:
            st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown("""
        ### 👨‍💻 Author
        **Agus Hartono** — NIM: E1E124025
        Machine Learning Project — Streamlit Deployment
        """)

# ══════════════════════════════════════
# FOOTER
# ══════════════════════════════════════
st.markdown("""
<div class="footer">
    💎 Diamond Price AI &nbsp;|&nbsp; Agus Hartono (E1E124025) &nbsp;|&nbsp;
    Machine Learning Project — Streamlit
</div>
""", unsafe_allow_html=True)
