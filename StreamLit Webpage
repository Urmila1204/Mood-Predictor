import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================
# Initialize dataset in session state
# =====================
if "df" not in st.session_state:
    data = {
        "Energy": ["High", "Medium", "Low", "High", "Low", "Medium", "High", "Low", "Medium", "High"],
        "Thoughts": ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive"],
        "Activity": ["Socialize", "Relax", "Work", "Socialize", "Relax", "Work", "Socialize", "Relax", "Work", "Socialize"],
        "Mood": ["Excited", "Sad", "Neutral", "Happy", "Sad", "Relaxed", "Excited", "Sad", "Neutral", "Happy"]
    }
    st.session_state.df = pd.DataFrame(data)

# Function to train model
def train_model(df):
    X = df[["Energy", "Thoughts", "Activity"]]
    y = df["Mood"]

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier()
    clf3 = GradientBoostingClassifier()

    voting_clf = VotingClassifier(estimators=[("lr", clf1), ("rf", clf2), ("gb", clf3)], voting="hard")
    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return voting_clf, encoder, acc, y_test, y_pred

# Train on current dataset
voting_clf, encoder, acc, y_test, y_pred = train_model(st.session_state.df)

# =====================
# Streamlit UI
# =====================
st.title("🎯 Mood Predictor App")
st.write("Enter your details to predict your mood or add a new entry to retrain the model.")

energy = st.text_input("Energy (High / Medium / Low)")
thoughts = st.text_input("Thoughts (Positive / Negative / Neutral)")
activity = st.text_input("Activity (Socialize / Relax / Work)")
new_mood = st.text_input("Mood (if adding a new entry)")

# Prediction Button
if st.button("✨ Predict Mood"):
    if energy and thoughts and activity:
        new_input = pd.DataFrame([[energy, thoughts, activity]], columns=["Energy", "Thoughts", "Activity"])
        try:
            new_input_encoded = encoder.transform(new_input).toarray()
            prediction = voting_clf.predict(new_input_encoded)[0]
            st.success(f"🎉 Predicted Mood: **{prediction}**")
        except:
            st.error("⚠️ Invalid input! Please use one of the available options.")
    else:
        st.warning("Please fill all the fields.")

# Add & Retrain Button
if st.button("➕ Add & Retrain"):
    if energy and thoughts and activity and new_mood:
        new_row = pd.DataFrame([[energy, thoughts, activity, new_mood]], columns=["Energy", "Thoughts", "Activity", "Mood"])
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        voting_clf, encoder, acc, y_test, y_pred = train_model(st.session_state.df)
        st.success("✅ New entry added and model retrained!")
    else:
        st.warning("Please fill all fields including the new mood.")

# =====================
# Show performance
# =====================
st.subheader("📊 Model Performance")
st.write(f"✅ Accuracy: {acc:.2f}")

st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Show dataset
st.subheader("📂 Current Dataset")
st.dataframe(st.session_state.df)

# Option to download dataset
csv = st.session_state.df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download Dataset as CSV", data=csv, file_name="meaningful_mood_dataset.csv", mime="text/csv")
