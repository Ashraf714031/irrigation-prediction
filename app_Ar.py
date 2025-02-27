import streamlit as st
import numpy as np
import joblib
from tensorflow import keras

# تحميل النموذج المحفوظ
lstm_model = keras.models.load_model("irrigation_time_lstm.h5")
scaler_irrigation = joblib.load("scaler_irrigation.pkl")

# قائمة التوصيات باللغة العربية
time_intervals = ["الآن", "بعد 6 ساعات", "بعد 12 ساعة", "بعد 18 ساعة", "بعد 24 ساعة",
                  "بعد 36 ساعة", "بعد 48 ساعة", "بعد 60 ساعة", "بعد 72 ساعة", "بعد 96 ساعة"]

# دالة التنبؤ بوقت الري
def predict_irrigation(features_values):
    features_values = np.array(features_values).reshape(1, -1)  # تحويل الإدخالات إلى مصفوفة NumPy
    features_scaled = scaler_irrigation.transform(features_values)  # تطبيق التحجيم
    input_sequence = features_scaled.reshape(1, 1, len(features_values[0]))  # إعادة تشكيل البيانات للنموذج
    prediction = lstm_model.predict(input_sequence)  # إجراء التنبؤ
    return time_intervals[np.argmax(prediction, axis=1)[0]]  # إرجاع التوصية المناسبة

# واجهة التطبيق باستخدام Streamlit
st.set_page_config(page_title="منظومة الري الذكي", page_icon="🌿", layout="centered")

st.title("🌿 منظومة الري الذكي")
st.write("أدخل بيانات الطقس وتوقعات المطر لمعرفة الوقت الأمثل للري.")

# إدخال بيانات المستخدم
sh = st.number_input("💧 رطوبة التربة (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
humidity = st.number_input("🌤️ الرطوبة الجوية (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
rainfall = st.number_input("🌧️ توقعات هطول الأمطار (مم)", min_value=0.0, max_value=500.0, value=5.0, step=0.1)

# زر التنبؤ
if st.button("🔍 توقع وقت الري"):
    result = predict_irrigation([sh, humidity, rainfall])
    st.success(f"⏳ التوصية: {result}")

# حقوق الملكية
st.markdown("---")
st.caption("© 2025 جميع الحقوق محفوظة | مالك أشرف أحمد صادق")
