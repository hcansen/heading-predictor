import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Başlık ve stil
st.markdown("<h1 style='color:#0e1117; font-size:42px;'>⛵︎ Gemi Heading Tahmini Uygulaması</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #444;'>", unsafe_allow_html=True)

# Model yükle
model_path = os.path.abspath("random_forest_heading_model_yeni.pkl")
model = joblib.load(model_path)

# ◉ Anlık Heading Tahmini
st.markdown("### ◉ Anlık Heading Tahmini")
st.markdown("<div style='color:#6c757d; font-size:14px;'>Rudder ve hız değerlerini girerek tahmini anında görün.</div>", unsafe_allow_html=True)

rudder = st.slider("◆ Rudder (derece)", -35.0, 35.0, 0.0)
speed = st.slider("◆ Hız (m/s)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

if st.button("▶︎ Tahmin Et"):
    input_data = np.array([[rudder, speed]])
    prediction = model.predict(input_data)[0] % 360
    st.success(f"⊕ Tahmin Edilen Heading Açısı: **{prediction:.2f}°**")

    fig, ax = plt.subplots()
    ax.scatter(1, prediction, color='#ff4b4b', s=120, label=f'{prediction:.2f}°')
    ax.set_ylim(0, 360)
    ax.set_xlim(0, 2)
    ax.set_title("⊘ Tahmin Noktası")
    ax.set_ylabel("Heading (derece)")
    ax.set_xticks([])
    ax.legend()
    st.pyplot(fig)

# ◉ CSV Yükleyerek Toplu Tahmin
st.markdown("<hr style='border:1px solid #444;'>", unsafe_allow_html=True)
st.markdown("### ◉ CSV Yükleyerek Toplu Heading Tahmini")
st.markdown("<div style='color:#6c757d; font-size:14px;'>CSV dosyası yükleyin (rudder, speed_total sütunları içermeli)</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📥 Dosya seçin", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'rudder' in data.columns and 'speed_total' in data.columns:
            input_data = data[['rudder', 'speed_total']].astype(float)
            predictions = model.predict(input_data) % 360
            data['Predicted_Heading'] = predictions

            st.markdown("#### ✦ Tahmin Tablosu")
            st.dataframe(data)

            st.markdown("#### ✦ İlk 100 Tahminin Grafiği")
            st.line_chart(data['Predicted_Heading'].head(100))

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Tahmin Sonuçlarını İndir", data=csv, file_name="tahmin_sonuclari.csv")
        else:
            st.error("❌ 'rudder' ve 'speed_total' sütunları bulunamadı.")
    except Exception as e:
        st.error(f"⚠️ Hata: {str(e)}")
