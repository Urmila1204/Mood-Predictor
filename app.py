import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
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
# Step 2: Train Model Function
# -----------------------------
def train_model(df):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier()
    clf3 = GradientBoostingClassifier()
    clf4 = SVC(probability=True)

    ensemble = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4)
    ], voting='soft')

    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, X_test, y_test, y_pred

# Initial training
ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, X_test, y_test, y_pred = train_model(df)

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
        ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, X_test, y_test, y_pred = train_model(df)

        st.success(f"✅ New mood '{new_mood}' added permanently and model retrained!")
    else:
        st.warning("Please fill all fields including the new mood!")

# Performance
st.subheader("📊 Model Performance")
st.write(f"✅ Accuracy: {acc:.2f}")

st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Show current dataset
st.subheader("📂 Current Dataset")
st.dataframe(df)

# Download dataset
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Dataset as CSV", data=csv, file_name="meaningful_mood_dataset.csv", mime="text/csv")
