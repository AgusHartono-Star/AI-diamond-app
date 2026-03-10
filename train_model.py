# train_model.py
# Jalankan file ini SEKALI untuk melatih dan menyimpan model XGBoost terbaik.
# python train_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

print("📂 Memuat dataset diamonds.csv ...")
df = pd.read_csv("diamonds.csv")
df = df.drop_duplicates()
print(f"✅ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

# ── Definisi Urutan Kategori (Ordinal) ──
cut_categories       = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_categories     = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_categories   = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

numerical_features   = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color', 'clarity']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OrdinalEncoder(categories=[
        cut_categories, color_categories, clarity_categories
    ]), categorical_features)
])

X = df.drop('price', axis=1)
y = df['price']

# ── Training semua model pada 4 skenario split ──
test_sizes = [0.90, 0.80, 0.70, 0.60]
models = {
    'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
}

full_results = []
best_r2 = -1
best_model_pipeline = None
best_model_name = None

print("\n🚀 Memulai Training semua model ...")
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    scenario_label = f"{int((1 - test_size) * 100)}:{int(test_size * 100)}"

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('prep', preprocessor),
            ('reg', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        full_results.append({
            'Model': name,
            'Skenario (Train:Test)': scenario_label,
            'R2 Score': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2)
        })
        print(f"  [{scenario_label}] {name:15s} → R²={r2:.4f}  RMSE={rmse:.0f}  MAE={mae:.0f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_pipeline = pipeline
            best_model_name = name

# ── Simpan hasil dan model terbaik ──
results_df = pd.DataFrame(full_results).sort_values('R2 Score', ascending=False)
results_df.to_csv("model_results.csv", index=False)

joblib.dump(best_model_pipeline, 'model_diamonds_final.pkl')

print(f"\n🏆 Model terbaik : {best_model_name} (R² = {best_r2:.4f})")
print("💾 Model disimpan : model_diamonds_final.pkl")
print("📊 Hasil evaluasi : model_results.csv")
