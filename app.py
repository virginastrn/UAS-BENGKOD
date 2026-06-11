import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model dan scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Prediksi Churn Pelanggan')
st.write('Aplikasi ini memprediksi apakah pelanggan akan churn (berhenti) atau tidak.')

# Form Input Sederhana (Sesuaikan dengan fitur penting dari hasil Feature Selection)
age = st.number_input('Usia', min_value=18, max_value=100)
total_spent = st.number_input('Total Pengeluaran ($)', min_value=0.0)
satisfaction_score = st.slider('Skor Kepuasan', 1.0, 5.0, 3.0)
# ... Tambahkan input lain sesuai jumlah fitur (X) yang digunakan saat training

if st.button('Prediksi Churn'):
    # Susun input menjadi dataframe/array sesuai urutan fitur saat ditraining
    input_data = np.array([[age, total_spent, satisfaction_score, ... ]]) # Lengkapi
    
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error('Pelanggan ini Berpotensi CHURN.')
    else:
        st.success('Pelanggan ini kemungkinan akan TETAP BERLANGGANAN.')