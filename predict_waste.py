import pandas as pd
import math

from sklearn.ensemble import RandomForestRegressor


# --------------------------------
# 1. LOAD DATA
# --------------------------------

df = pd.read_csv("data/austin_waste.csv")

df["Report Date"] = pd.to_datetime(df["Report Date"])

df = df.dropna(subset=["Load Weight"])


# --------------------------------
# 2. CREATE DAILY WASTE DATA
# --------------------------------

daily_waste = (
    df.groupby("Report Date")["Load Weight"]
    .sum()
    .reset_index()
)

daily_waste.columns = ["Date", "Total_Waste"]

daily_waste = daily_waste.sort_values("Date").reset_index(drop=True)


# --------------------------------
# 3. CREATE FEATURES
# --------------------------------

daily_waste["Day_of_Week"] = daily_waste["Date"].dt.dayofweek
daily_waste["Month"] = daily_waste["Date"].dt.month
daily_waste["Day"] = daily_waste["Date"].dt.day

daily_waste["Previous_Day_Waste"] = (
    daily_waste["Total_Waste"].shift(1)
)

daily_waste["Previous_7_Day_Avg"] = (
    daily_waste["Total_Waste"]
    .shift(1)
    .rolling(7)
    .mean()
)

daily_waste = daily_waste.dropna().reset_index(drop=True)


# --------------------------------
# 4. SELECT FEATURES
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
# 5. TRAIN RANDOM FOREST
# --------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# --------------------------------
# 6. GET PREDICTION DATE
# --------------------------------

date_input = input(
    "\nEnter prediction date (YYYY-MM-DD): "
)

prediction_date = pd.to_datetime(date_input)


# --------------------------------
# 7. GET RECENT WASTE DATA
# --------------------------------

previous_day_waste = daily_waste["Total_Waste"].iloc[-1]

previous_7_day_avg = (
    daily_waste["Total_Waste"]
    .tail(7)
    .mean()
)


# --------------------------------
# 8. CREATE INPUT FOR MODEL
# --------------------------------

input_data = pd.DataFrame({
    "Day_of_Week": [prediction_date.dayofweek],
    "Month": [prediction_date.month],
    "Day": [prediction_date.day],
    "Previous_Day_Waste": [previous_day_waste],
    "Previous_7_Day_Avg": [previous_7_day_avg]
})


# --------------------------------
# 9. PREDICT WASTE
# --------------------------------

prediction = model.predict(input_data)

predicted_waste = prediction[0]


print("\n==============================")
print("WASTE PREDICTION")
print("==============================")

print(
    "Prediction Date:",
    prediction_date.date()
)

print(
    "Predicted Waste:",
    round(predicted_waste, 2)
)


# --------------------------------
# 10. COLLECTION CAPACITY
# --------------------------------

truck_capacity = float(
    input("\nEnter truck capacity: ")
)

required_trips = math.ceil(
    predicted_waste / truck_capacity
)


print("\n==============================")
print("COLLECTION REQUIREMENT")
print("==============================")

print(
    "Truck Capacity:",
    truck_capacity
)

print(
    "Required Trips:",
    required_trips
)

print("\nPrediction completed successfully!")