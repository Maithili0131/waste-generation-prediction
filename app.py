import streamlit as st
import pandas as pd
import math
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Waste Prediction System",
    page_icon="♻️",
    layout="centered"
)


# --------------------------------
# TITLE
# --------------------------------

st.title("♻️ Waste Generation Prediction System")

st.write(
    "Predict upcoming waste generation using historical "
    "waste collection data and estimate the required "
    "collection trips."
)


# --------------------------------
# LOAD DATA
# --------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/austin_waste.csv")

    df["Report Date"] = pd.to_datetime(
        df["Report Date"]
    )

    df = df.dropna(
        subset=["Load Weight"]
    )

    daily_waste = (
        df.groupby("Report Date")["Load Weight"]
        .sum()
        .reset_index()
    )

    daily_waste.columns = [
        "Date",
        "Total_Waste"
    ]

    daily_waste = (
        daily_waste
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return daily_waste


daily_waste = load_data()


# --------------------------------
# FEATURE ENGINEERING
# --------------------------------

daily_waste["Day_of_Week"] = (
    daily_waste["Date"].dt.dayofweek
)

daily_waste["Month"] = (
    daily_waste["Date"].dt.month
)

daily_waste["Day"] = (
    daily_waste["Date"].dt.day
)

daily_waste["Previous_Day_Waste"] = (
    daily_waste["Total_Waste"].shift(1)
)

daily_waste["Previous_7_Day_Avg"] = (
    daily_waste["Total_Waste"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily_waste = (
    daily_waste
    .dropna()
    .reset_index(drop=True)
)


# --------------------------------
# FEATURES AND TARGET
# --------------------------------

features = [
    "Day_of_Week",
    "Month",
    "Day",
    "Previous_Day_Waste",
    "Previous_7_Day_Avg"
]

X = daily_waste[features]

y = daily_waste["Total_Waste"]


# --------------------------------
# TIME-BASED TRAIN / TEST SPLIT
# --------------------------------

split_index = int(len(daily_waste) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# --------------------------------
# TRAIN MODEL FOR EVALUATION
# --------------------------------

evaluation_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

evaluation_model.fit(
    X_train,
    y_train
)


# --------------------------------
# MODEL EVALUATION
# --------------------------------

y_pred = evaluation_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# --------------------------------
# FINAL MODEL
# --------------------------------
# Train on all available historical
# data for future predictions.

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X,
    y
)


# --------------------------------
# USER INPUT
# --------------------------------

st.subheader("Prediction Details")

prediction_date = st.date_input(
    "Select Prediction Date"
)

truck_capacity = st.number_input(
    "Enter Truck Capacity",
    min_value=1.0,
    value=400000.0,
    step=1000.0
)


# --------------------------------
# PREDICTION
# --------------------------------

if st.button("🔮 Predict Waste"):

    prediction_date = pd.Timestamp(
        prediction_date
    )

    selected_day = (
        prediction_date.dayofweek
    )


    # --------------------------------
    # RECENT 30-DAY AVERAGE
    # --------------------------------

    recent_average = (
        daily_waste["Total_Waste"]
        .tail(30)
        .mean()
    )


    # --------------------------------
    # SAME DAY-OF-WEEK AVERAGE
    # --------------------------------

    same_day_data = daily_waste[
        daily_waste["Day_of_Week"] == selected_day
    ]

    same_day_average = (
        same_day_data["Total_Waste"]
        .tail(20)
        .mean()
    )


    # --------------------------------
    # PREVIOUS 7-DAY AVERAGE
    # --------------------------------

    previous_7_day_avg = (
        daily_waste["Total_Waste"]
        .tail(7)
        .mean()
    )


    # --------------------------------
    # ACTUAL PREVIOUS DAY WASTE
    # --------------------------------

    previous_day_waste = (
        daily_waste["Total_Waste"].iloc[-1]
    )


    # --------------------------------
    # MODEL INPUT
    # --------------------------------

    input_data = pd.DataFrame({

        "Day_of_Week": [
            prediction_date.dayofweek
        ],

        "Month": [
            prediction_date.month
        ],

        "Day": [
            prediction_date.day
        ],

        "Previous_Day_Waste": [
            previous_day_waste
        ],

        "Previous_7_Day_Avg": [
            previous_7_day_avg
        ]
    })


    # --------------------------------
    # RANDOM FOREST PREDICTION
    # --------------------------------

    model_prediction = model.predict(
        input_data
    )[0]


    # --------------------------------
    # FINAL COMBINED PREDICTION
    # --------------------------------

    predicted_waste = (
        model_prediction * 0.70
        + same_day_average * 0.20
        + recent_average * 0.10
    )


    # --------------------------------
    # COLLECTION TRIPS
    # --------------------------------

    required_trips = math.ceil(
        predicted_waste /
        truck_capacity
    )


    # --------------------------------
    # DISPLAY RESULT
    # --------------------------------

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Waste",
            f"{predicted_waste:,.2f}"
        )

    with col2:

        st.metric(
            "Required Trips",
            required_trips
        )


    st.success(
        f"For {prediction_date.date()}, "
        f"the predicted waste generation is "
        f"{predicted_waste:,.2f}. "
        f"Approximately {required_trips} "
        f"collection trips are required."
    )


    # --------------------------------
    # CALCULATION DETAILS
    # --------------------------------

    st.subheader("Prediction Details")

    st.write(
        f"Random Forest prediction: "
        f"{model_prediction:,.2f}"
    )

    st.write(
        f"Same day-of-week average: "
        f"{same_day_average:,.2f}"
    )

    st.write(
        f"Recent 30-day average: "
        f"{recent_average:,.2f}"
    )

    st.info(
        "Final prediction combines the Random Forest "
        "prediction with historical weekly and recent "
        "waste patterns."
    )


# --------------------------------
# MODEL PERFORMANCE
# --------------------------------

st.divider()

st.subheader("🤖 Model Performance")

st.write(
    "The following metrics are calculated on the "
    "20% chronological test set."
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )

with col2:

    st.metric(
        "MAE",
        f"{mae:,.2f}"
    )

with col3:

    st.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

st.caption(
    "R², MAE and RMSE evaluate the Random Forest model "
    "on previously unseen historical data. The final "
    "dashboard prediction additionally combines the "
    "model output with historical averages."
)


# --------------------------------
# HISTORICAL DATA
# --------------------------------

st.divider()

st.subheader("📊 Historical Waste Generation")

chart_data = daily_waste[
    ["Date", "Total_Waste"]
].set_index("Date")

st.line_chart(
    chart_data
)


# --------------------------------
# DATASET INFORMATION
# --------------------------------

st.divider()

st.subheader("📁 Dataset Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Daily Records",
        len(daily_waste)
    )

with col2:

    st.metric(
        "Model",
        "Random Forest"
    )


st.caption(
    "The prediction is based on historical municipal "
    "waste collection data."
)