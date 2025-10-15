import streamlit as st
import pandas as pd
import pickle
import os

# Load model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/xgb_nfl_model.pkl"))
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please train and save the model first.")
    st.stop()
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🏈 NFL Game Outcome Predictor", layout="centered")
st.title("🏈 NFL Game Outcome Predictor")
st.write("Predict whether a team will win or lose a game based on historical stats.")

# Sidebar for input
st.sidebar.header("Input Game Stats")
def user_input_features():
    plays = st.sidebar.number_input("Total Plays", 40, 120, 65)
    total_yards = st.sidebar.number_input("Total Yards Gained", 100, 600, 350)
    yards_per_play = st.sidebar.number_input("Yards per Play", 3.0, 10.0, 5.5, step=0.1)
    rush_pass_ratio = st.sidebar.number_input("Rush/Pass Ratio", 0.0, 1.0, 0.45, step=0.01)
    turnover_diff = st.sidebar.number_input("Total Turnovers", 0, 10, 1)
    avg_epa = st.sidebar.number_input("Average EPA", -2.0, 2.0, 0.05, step=0.01)
    fg_made = st.sidebar.number_input("Field Goals Made", 0, 5, 2)
    fg_missed = st.sidebar.number_input("Field Goals Missed", 0, 5, 0)
    first_downs = st.sidebar.number_input("First Downs", 0, 30, 15)
    
    data = {
        "plays": plays,
        "total_yards": total_yards,
        "yards_per_play": yards_per_play,
        "rush_pass_ratio": rush_pass_ratio,
        "turnover_diff": turnover_diff,
        "avg_epa": avg_epa,
        "fg_made": fg_made,
        "fg_missed": fg_missed,
        "first_downs": first_downs
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
try:
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

st.subheader("Prediction")
st.write("🏆 Win" if prediction == 1 else "❌ Loss")

st.subheader("Prediction Probability")
st.write(f"Win Probability: {prediction_proba[1]*100:.2f}%")
st.write(f"Loss Probability: {prediction_proba[0]*100:.2f}%")
