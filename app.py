import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="E-Commerce Purchase Predictor",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 E-Commerce Purchase Intent Predictor")
st.write("Predict whether an e-commerce session is likely to result in a purchase.")

# Load dataset
DATA_PATH = "data/ecommerce_data.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found at: {DATA_PATH}")
    st.stop()

features = [
    "pages_viewed",
    "session_minutes",
    "products_viewed",
    "cart_additions",
    "discount_seen",
    "previous_orders"
]

X = df[features]
y = df["target"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        solver="liblinear",
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

st.subheader("Enter Session Information")

pages_viewed = st.number_input(
    "Pages Viewed",
    min_value=0,
    value=5
)

session_minutes = st.number_input(
    "Session Duration (minutes)",
    min_value=0.0,
    value=10.0
)

products_viewed = st.number_input(
    "Products Viewed",
    min_value=0,
    value=3
)

cart_additions = st.number_input(
    "Cart Additions",
    min_value=0,
    value=1
)

discount_seen = st.number_input(
    "Discount / Promotion Exposure",
    min_value=0.0,
    value=1.0
)

previous_orders = st.number_input(
    "Previous Orders",
    min_value=0,
    value=0
)

input_data = pd.DataFrame([{
    "pages_viewed": pages_viewed,
    "session_minutes": session_minutes,
    "products_viewed": products_viewed,
    "cart_additions": cart_additions,
    "discount_seen": discount_seen,
    "previous_orders": previous_orders
}])

if st.button("Predict Purchase", type="primary"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🛍️ Purchase Likely")
    else:
        st.warning("❌ Purchase Not Likely")

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )