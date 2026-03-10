# 💎 Diamond Price AI

Aplikasi web Machine Learning untuk memprediksi harga berlian berdasarkan karakteristik fisiknya, dibangun dengan **Streamlit** dan **XGBoost**.

**Dibuat oleh:** Agus Hartono — NIM: E1E124025

---

## 🚀 Cara Menjalankan

### 1. Install dependensi
```bash
pip install -r requirements.txt
```

### 2. Latih model (jalankan sekali)
```bash
python train_model.py
```

### 3. Jalankan aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`

---

## 📦 Struktur Proyek

```
AI-diamond-app/
├── app.py                    # Aplikasi Streamlit utama
├── train_model.py            # Script training & simpan model
├── style.css                 # Custom CSS
├── requirements.txt          # Dependensi Python
├── diamonds.csv              # Dataset
├── model_diamonds_final.pkl  # Model XGBoost (setelah training)
├── model_results.csv         # Hasil evaluasi (setelah training)
└── e1e124025_agus_hartono.py # Kode asli Google Colab
```

---

## 🤖 Model Machine Learning

| Model | Deskripsi |
|---|---|
| K-Nearest Neighbors | Instance-based learning |
| Random Forest | Ensemble bagging |
| **XGBoost ✅** | Gradient boosting — **akurasi tertinggi** |

**Metrik Evaluasi:** R² Score, RMSE, MAE  
**Skenario Split:** 10:90, 20:80, 30:70, 40:60

---

## 🔮 Fitur Aplikasi

- **Tab Prediksi** — Input fitur berlian, prediksi harga real-time + rentang estimasi
- **Tab EDA** — Eksplorasi dataset (distribusi, korelasi, scatter plot)
- **Tab Evaluasi** — Perbandingan 3 model di 4 skenario split
- **Tab Tentang** — Deskripsi proyek dan tech stack
