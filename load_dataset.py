import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------
# 1. LOAD DATA
# --------------------------------

df = pd.read_csv("data/austin_waste.csv")

print("Original dataset shape:")
print(df.shape)

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

daily_waste = (
    daily_waste
    .sort_values("Date")
    .reset_index(drop=True)
)

# --------------------------------
# 3. FEATURE ENGINEERING
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
# 5. TIME-BASED TRAIN/TEST SPLIT
# --------------------------------

split_index = int(len(daily_waste) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------
# 6. TRAIN RANDOM FOREST
# --------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------
# 7. PREDICT TEST DATA
# --------------------------------

y_pred = model.predict(X_test)

# --------------------------------
# 8. MODEL EVALUATION
# --------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# --------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# --------------------------------
# 10. CHECK WEEKLY WASTE PATTERN
# --------------------------------

weekly_pattern = (
    daily_waste
    .groupby("Day_of_Week")["Total_Waste"]
    .agg(["mean", "min", "max", "count"])
)

day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

weekly_pattern.index = weekly_pattern.index.map(day_names)

print("\n==============================")
print("WEEKLY WASTE PATTERN")
print("==============================")

print(weekly_pattern.round(2))

# --------------------------------
# 11. COMPARE MODEL WITH/WITHOUT
#     DAY OF WEEK
# --------------------------------

features_without_day = [
    "Month",
    "Day",
    "Previous_Day_Waste",
    "Previous_7_Day_Avg"
]

X2 = daily_waste[features_without_day]
y2 = daily_waste["Total_Waste"]

X2_train = X2.iloc[:split_index]
X2_test = X2.iloc[split_index:]

y2_train = y2.iloc[:split_index]
y2_test = y2.iloc[split_index:]

model_without_day = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model_without_day.fit(X2_train, y2_train)

y2_pred = model_without_day.predict(X2_test)

mae2 = mean_absolute_error(y2_test, y2_pred)
mse2 = mean_squared_error(y2_test, y2_pred)
rmse2 = np.sqrt(mse2)
r2_2 = r2_score(y2_test, y2_pred)

print("\n==============================")
print("MODEL WITHOUT DAY_OF_WEEK")
print("==============================")

print(f"MAE  : {mae2:.2f}")
print(f"MSE  : {mse2:.2f}")
print(f"RMSE : {rmse2:.2f}")
print(f"R²   : {r2_2:.4f}")