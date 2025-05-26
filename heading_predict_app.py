import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Başlık
st.title("🧭 Gemi Heading Tahmin Uygulaması")

# Modeli yükle
model_path = os.path.abspath("random_forest_heading_model_yeni.pkl")
model = joblib.load(model_path)

# 🔹 Tekli Tahmin Girişi
st.header("🎯 Elle Heading Tahmini")
rudder = st.slider("Rudder (derece)", -35.0, 35.0, 0.0)
speed = st.slider("Speed (m/s)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

if st.button("Tahmin Et"):
    input_data = np.array([[rudder, speed]])
    raw_prediction = model.predict(input_data)[0]
    prediction = raw_prediction % 360  # Normalize 0–360

    st.success(f"🔍 Tahmin Edilen Heading Açısı: {prediction:.2f}°")

    # Grafikle göster
    fig, ax = plt.subplots()
    ax.scatter(1, prediction, color='red', s=100, label=f'{prediction:.2f}°')
    ax.set_ylim(0, 360)
    ax.set_xlim(0, 2)
    ax.set_title("Tahmin Heading Açısı")
    ax.set_ylabel("Heading (derece)")
    ax.set_xticks([])
    ax.legend()
    st.pyplot(fig)

# 🔹 CSV Yükleme ve Toplu Tahmin
st.markdown("---")
st.header("📁 CSV Yükleyerek Toplu Heading Tahmini")

uploaded_file = st.file_uploader("CSV dosyasını yükleyin (rudder, speed_total içermeli)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Dosya başarıyla yüklendi.")

        # Giriş sütunları kontrolü
        if 'rudder' in data.columns and 'speed_total' in data.columns:
            input_features = data[['rudder', 'speed_total']].astype(float)
            predictions = model.predict(input_features)
            predictions = predictions % 360
            data['Predicted_Heading'] = predictions

            # Tablo göster
            st.subheader("📋 Tahmin Sonuçları")
            st.dataframe(data)

            # Grafik göster (ilk 100 tahmin)
            st.subheader("📈 Tahmin Grafiği (İlk 100)")
            st.line_chart(data['Predicted_Heading'].head(100))

            # İndirme
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Tahminleri CSV olarak indir", data=csv, file_name="heading_tahmin_sonuclari.csv")

        else:
            st.error("❌ CSV'de 'rudder' ve 'speed_total' sütunları bulunmalı.")
    except Exception as e:
        st.error(f"❌ Hata: {str(e)}")
