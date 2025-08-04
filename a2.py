



# Define the tokenizer used in TF-IDF Vectorizer
def comma_tokenizer(x):
    return x.split(',')

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Set config as the first Streamlit command
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Load model
model = joblib.load("car_price_predictor.pkl")

# Define options from dataset (your unique lists)
make_model_options = [
    'Audi A1', 'Audi A2', 'Audi A3', 'Opel Astra', 'Opel Corsa',
    'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'
]
body_type_options = [
    'Sedans', 'Station wagon', 'Compact', 'Van', 'Transporter', 'Off-Road', 'Coupe', 'Convertible'
]
vat_options = ['VAT deductible', 'Price negotiable']
type_options = ['Used', 'New', 'Pre-registered', "Employee's car", 'Demonstration']
fuel_options = ['Benzine', 'Diesel', 'LPG/CNG', 'Electric']
paint_options = ['Metallic', 'Uni/basic', 'Perl effect']
upholstery_options = ['Cloth', 'Part/Full Leather']
gearing_options = ['Manual', 'Automatic', 'Semi-automatic']
drive_chain_options = ['front', '4WD', 'rear']

comfort_options = ['Air conditioning', 'Power windows', 'Electrical side mirrors',
    'Multi-function steering wheel', 'Cruise control', 'Park Distance Control',
    'Parking assist system sensors rear', 'Leather steering wheel', 'Start-stop system',
    'Automatic climate control', 'Rain sensor', 'Navigation system', 'Light sensor',
    'Armrest', 'Seat heating', 'Hill Holder', 'Parking assist system sensors front',
    'Parking assist system camera', 'Lumbar support', 'Heated steering wheel',
    'Keyless central door lock', 'Split rear seats', 'Electrically adjustable seats',
    'Tinted windows', 'Electric tailgate', 'Electrically heated windshield', 'Seat ventilation',
    'Parking assist system self-steering', 'Panorama roof', 'Heads-up display', 'Sunroof',
    'Massage seats', 'Auxiliary heating', 'Air suspension', 'Leather seats', 'Wind deflector',
    'Windshield', 'Electric Starter']

entertainment_options = ['On-board computer', 'Radio', 'Bluetooth', 'Hands-free equipment',
    'USB', 'MP3', 'CD player', 'Sound system', 'Digital radio', 'Television']

extras_options = ['Alloy wheels', 'Voice Control', 'Touch screen', 'Sport seats', 'Roof rack',
    'Catalytic Converter', 'Sport suspension', 'Sport package', 'Trailer hitch', 'Shift paddles',
    'Cab or rented Car', 'Ski bag', 'Winter tyres', 'Handicapped enabled', 'Tuned car',
    'Sliding door', 'Right hand drive']

safety_options = ['ABS', 'Driver-side airbag', 'Power steering', 'Passenger-side airbag',
    'Electronic stability control', 'Side airbag', 'Central door lock', 'Isofix',
    'Traction control', 'Tire pressure monitoring system', 'Daytime running lights',
    'Immobilizer', 'Fog lights', 'LED Daytime Running Lights', 'Xenon headlights',
    'Emergency brake assistant', 'LED Headlights', 'Lane departure warning system',
    'Central door lock with remote control', 'Head airbag', 'Adaptive headlights',
    'Traffic sign recognition', 'Alarm system', 'Emergency system', 'Blind spot monitor',
    'Adaptive Cruise Control', 'Rear airbag', 'Driver drowsiness detection', 'Night view assist']

# Sidebar UI
st.sidebar.title("Select Your Dream Car Features")

st.markdown("---")
st.markdown("✨ Made with ❤️ by **Zafer**")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 App by **Zafer Yildirim**")

make_model = st.sidebar.selectbox("Make & Model", make_model_options)
body_type = st.sidebar.selectbox("Body Type", body_type_options)
vat = st.sidebar.selectbox("VAT Option", vat_options)
car_type = st.sidebar.selectbox("Car Type", type_options)
fuel = st.sidebar.selectbox("Fuel Type", fuel_options)
paint = st.sidebar.selectbox("Paint Type", paint_options)
upholstery = st.sidebar.selectbox("Upholstery Type", upholstery_options)
gear_type = st.sidebar.selectbox("Gearing Type", gearing_options)
drive_chain = st.sidebar.selectbox("Drive Chain", drive_chain_options)

# Numeric inputs
km = st.sidebar.number_input("Kilometres", min_value=0.0, value=25000.0)
gears = st.sidebar.number_input("Number of Gears", min_value=5.0, max_value=8.0, value=6.0)
age = st.sidebar.number_input("Car Age (Years)", min_value=0.0, max_value=3.0, value=1.0)
owners = st.sidebar.number_input("Previous Owners", min_value=0.0, max_value=4.0, value=1.0)
hp = st.sidebar.number_input("Horsepower (kW)", min_value=40.0, max_value=239.0, value=90.0)
inspection = st.sidebar.selectbox("Inspection New", [0, 1])
displacement = st.sidebar.number_input("Displacement (cc)", min_value=890.0, max_value=2967.0, value=1400.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=840.0, max_value=2471.0, value=1300.0)
consumption = st.sidebar.number_input("Consumption Combined (L/100km)", min_value=3.0, max_value=9.1, value=5.0)

# Multi-select options (converted to comma-joined string)
comfort = st.sidebar.multiselect("Comfort & Convenience", comfort_options)
entertainment = st.sidebar.multiselect("Entertainment & Media", entertainment_options)
extras = st.sidebar.multiselect("Extras", extras_options)
safety = st.sidebar.multiselect("Safety & Security", safety_options)

# Build DataFrame for prediction
car_features = pd.DataFrame({
    'make_model': [make_model],
    'body_type': [body_type],
    'vat': [vat],
    'km': [km],
    'Type': [car_type],
    'Fuel': [fuel],
    'Gears': [gears],
    'Comfort_Convenience': [','.join(comfort)],
    'Entertainment_Media': [','.join(entertainment)],
    'Extras': [','.join(extras)],
    'Safety_Security': [','.join(safety)],
    'age': [age],
    'Previous_Owners': [owners],
    'hp_kW': [hp],
    'Inspection_new': [inspection],
    'Paint_Type': [paint],
    'Upholstery_type': [upholstery],
    'Gearing_Type': [gear_type],
    'Displacement_cc': [displacement],
    'Weight_kg': [weight],
    'Drive_chain': [drive_chain],
    'cons_comb': [consumption]
})

st.markdown("## Your Dream Car's Features")
st.write(car_features)


if st.button("Predict Price"):
    predicted_price = model.predict(car_features)[0]

    st.markdown(f"""
        <style>
        @keyframes slide-in {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        .prediction-box {{
            animation: slide-in 0.8s ease-out;
            background: #fff8f0;
            padding: 1.5em;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 6px solid #e53935;
            font-size: 18px;
            margin-top: 25px;
        }}
        .prediction-price {{
            color: #e53935;
            font-weight: bold;
            font-size: 22px;
        }}
        .prediction-call {{
            color: #1565c0;
            font-weight: 600;
            margin-top: 0.5em;
        }}
        </style>

        <div class="prediction-box">
            🚗 <span class="prediction-price">Predicted Dream Price: €{predicted_price:,.2f}</span><br>
            📞 <span class="prediction-call">Call us now to catch this <u>unmissable deal</u>!</span>
        </div>
    """, unsafe_allow_html=True)




# Optional image
try:
    image = Image.open("car1.jpg")
    st.image("car1.jpg", use_container_width=True)
except:
    st.warning("Car image not found. Add 'car1.jpg' in same folder for visual appeal.")
