# 💎 Diamond Price AI

A machine learning web application that predicts diamond prices based on their physical characteristics, built with Streamlit and XGBoost.

---

## 🚀 Live Demo

Coming soon (Streamlit Cloud deployment)

---

## ✨ Features

- Predict diamond prices using a trained XGBoost model
- Interactive sliders and dropdowns for all diamond features
- Feature importance visualization
- Clean, responsive Streamlit dashboard

---

## 🛠️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-diamond-app.git
cd AI-diamond-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📦 Project Structure

```
AI-diamond-app/
├── app.py                    # Main Streamlit application
├── style.css                 # Custom CSS styles
├── model_diamonds_final.pkl  # Trained ML model (XGBoost pipeline)
├── requirements.txt          # Python dependencies
├── assets/
│   ├── diamond_banner.png    # Hero banner image
│   └── ai.webp               # AI section image
└── README.md
```

---

## 🤖 Machine Learning Models

This project compares several regression models trained on the `diamonds` dataset (ggplot2):

| Model | Description |
|---|---|
| K-Nearest Neighbors (KNN) | Instance-based learning |
| Random Forest | Bagging ensemble |
| XGBoost ✅ | Gradient boosting (final model) |

**Evaluation Metrics:** R² Score, RMSE, MAE

---

## 🔮 Input Features

| Feature | Type | Description |
|---|---|---|
| Carat | Float | Weight of the diamond |
| Cut | Categorical | Quality of the cut (Fair → Ideal) |
| Color | Categorical | Diamond colour (J = worst, D = best) |
| Clarity | Categorical | Clarity grade (I1 → IF) |
| Depth | Float | Total depth percentage |
| Table | Float | Width of top of diamond |
| x, y, z | Float | Length, width, height (mm) |

---

## 📄 License

MIT License
