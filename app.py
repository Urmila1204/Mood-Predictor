# ================================
# 🔮 Mood Predictor with Streamlit UI
# ================================

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# -----------------------------
# Step 1: Load or Create Dataset
# -----------------------------
DATA_FILE = "mood_dataset.csv"

def load_dataset():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame({
            "Energy": ["High", "Medium", "Low", "High", "Low", "Medium", "High", "Low", "Medium", "High"],
            "Thoughts": ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive"],
            "Activity": ["Socialize", "Relax", "Work", "Relax", "Work", "Socialize", "Work", "Relax", "Socialize", "Work"],
            "Mood": ["Excited", "Sad", "Happy", "Relaxed", "Neutral", "Excited", "Sad", "Relaxed", "Neutral", "Happy"]
        })
        df.to_csv(DATA_FILE, index=False)
    return df

df = load_dataset()

# -----------------------------
# Step 2: Train Model Function (Full Dataset)
# -----------------------------
def train_model_full(df):
    le_energy = LabelEncoder()
    le_thoughts = LabelEncoder()
    le_activity = LabelEncoder()
    le_mood = LabelEncoder()

    df["Energy_enc"] = le_energy.fit_transform(df["Energy"])
    df["Thoughts_enc"] = le_thoughts.fit_transform(df["Thoughts"])
    df["Activity_enc"] = le_activity.fit_transform(df["Activity"])
    df["Mood_enc"] = le_mood.fit_transform(df["Mood"])

    X = df[["Energy_enc", "Thoughts_enc", "Activity_enc"]]
    y = df["Mood_enc"]

    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier()
    clf3 = GradientBoostingClassifier()
    clf4 = SVC(probability=True)

    ensemble = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4)
    ], voting='soft')

    ensemble.fit(X, y)

    y_pred = ensemble.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, zero_division=0, output_dict=False)

    return ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, cm, cr

ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, cm, cr = train_model_full(df)

# -----------------------------
# Step 3: Streamlit UI
# -----------------------------
st.title("🎭 Mood Predictor with Permanent Add & Retrain")

energy = st.text_input("Energy (High / Medium / Low):")
thoughts = st.text_input("Thoughts (Positive / Negative / Neutral):")
activity = st.text_input("Activity (Socialize / Relax / Work):")
new_mood = st.text_input("Mood (if adding a new entry):")

# Prediction
if st.button("✨ Predict Mood"):
    if energy and thoughts and activity:
        try:
            encoded_input = [
                le_energy.transform([energy])[0],
                le_thoughts.transform([thoughts])[0],
                le_activity.transform([activity])[0]
            ]
            prediction = ensemble.predict([encoded_input])[0]
            mood_label = le_mood.inverse_transform([prediction])[0]
            st.success(f"🎉 Predicted Mood: {mood_label}")
            st.info(f"📊 Current Model Accuracy (on full dataset): {acc:.2f}")
            st.text("Confusion Matrix:")
            st.write(cm)
            st.text("Classification Report:")
            st.text(cr)
        except:
            st.error("⚠️ Invalid input! Please add this combination first using Add & Retrain.")
    else:
        st.warning("Please fill all fields!")

# Add & Retrain
if st.button("➕ Add & Retrain"):
    if energy and thoughts and activity and new_mood:
        # Append new row permanently
        new_entry = pd.DataFrame([[energy, thoughts, activity, new_mood]],
                                 columns=["Energy", "Thoughts", "Activity", "Mood"])
        new_entry.to_csv(DATA_FILE, mode='a', header=False, index=False)

        # Reload dataset and retrain model immediately
        df = load_dataset()
        ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, cm, cr = train_model_full(df)

        st.success(f"✅ New mood '{new_mood}' added permanently and model retrained!")
        st.info(f"📊 Updated Model Accuracy: {acc:.2f}")
        st.text("Confusion Matrix:")
        st.write(cm)
        st.text("Classification Report:")
        st.text(cr)
    else:
        st.warning("Please fill all fields including the new mood!")

# Show current dataset
st.subheader("📂 Current Dataset")
st.dataframe(df)

# Download dataset
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Dataset as CSV", data=csv, file_name="mood_dataset.csv", mime="text/csv")
