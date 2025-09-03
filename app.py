# ================================
# 🔮 Mood Predictor with Risk Warnings
# ================================

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# -----------------------------
# Step 1: Load or Create Dataset
# -----------------------------
DATA_FILE = "meaningful_mood_dataset.csv"

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
    clf_knn = KNeighborsClassifier(n_neighbors=3)

    ensemble = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gb', clf3), ('svc', clf4), ('knn', clf_knn)
    ], voting='soft')

    ensemble.fit(X, y)

    y_pred = ensemble.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, zero_division=0, output_dict=False)

    return ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, cm, cr

ensemble, le_energy, le_thoughts, le_activity, le_mood, acc, cm, cr = train_model_full(df)

# -----------------------------
# Step 3: Risky Keywords
# -----------------------------
RISKY_KEYWORDS = ["suicidal", "depressed", "hopeless", "self-harm", "anxious", "overwhelmed"]

def check_risky_thoughts(thoughts_input):
    return any(word.lower() in thoughts_input.lower() for word in RISKY_KEYWORDS)

# -----------------------------
# Step 4: Streamlit UI
# -----------------------------
st.title("🎭 Mood Predictor with Risk Warnings & Auto-Prediction")

energy = st.text_input("Energy (High / Medium / Low):")
thoughts = st.text_input("Thoughts (Positive / Negative / Neutral / Others):")
activity = st.text_input("Activity (Socialize / Relax / Work):")
new_mood = st.text_input("Mood (if adding a new entry manually):")

# Prediction
if st.button("✨ Predict Mood"):
    if energy and thoughts and activity:
        try:
            # Encode inputs safely
            encoded_input = [
                le_energy.transform([energy])[0] if energy in le_energy.classes_ else -1,
                le_thoughts.transform([thoughts])[0] if thoughts in le_thoughts.classes_ else -1,
                le_activity.transform([activity])[0] if activity in le_activity.classes_ else -1
            ]

            # Predict using kNN if any input is unseen
            if -1 in encoded_input:
                knn = KNeighborsClassifier(n_neighbors=3)
                X = df[["Energy_enc", "Thoughts_enc", "Activity_enc"]]
                y = df["Mood_enc"]
                knn.fit(X, y)
                prediction = knn.predict([encoded_input])[0]
            else:
                prediction = ensemble.predict([encoded_input])[0]

            mood_label = le_mood.inverse_transform([prediction])[0]
            st.success(f"🎉 Predicted Mood: {mood_label}")

            # Check for risky thoughts
            if check_risky_thoughts(thoughts):
                st.warning(
                    "⚠️ It seems you are expressing risky or negative thoughts. "
                    "Please consider talking to a trusted friend, family member, or a mental health professional immediately."
                )
                st.markdown(
                    "📞 **Helplines:** If you feel unsafe, call 988 (India) or visit [Befrienders Worldwide](https://www.befrienders.org)"
                )

            st.info(f"📊 Current Model Accuracy: {acc:.2f}")
            st.text("Confusion Matrix:")
            st.write(cm)
            st.text("Classification Report:")
            st.text(cr)

        except Exception as e:
            st.error(f"⚠️ Something went wrong with the prediction! ({e})")
    else:
        st.warning("Please fill all fields!")

# Add & Retrain manually
if st.button("➕ Add & Retrain"):
    if energy and thoughts and activity and new_mood:
        new_entry = pd.DataFrame([[energy, thoughts, activity, new_mood]],
                                 columns=["Energy", "Thoughts", "Activity", "Mood"])
        new_entry.to_csv(DATA_FILE, mode='a', header=False, index=False)

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

# Show dataset
st.subheader("📂 Current Dataset")
st.dataframe(df)

# Download dataset
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Dataset as CSV", data=csv, file_name="meaningful_mood_dataset.csv", mime="text/csv")
