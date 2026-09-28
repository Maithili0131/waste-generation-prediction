import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. LOAD DATA

df = pd.read_csv("data/austin_waste.csv")

df["Report Date"] = pd.to_datetime(df["Report Date"])

df = df.dropna(subset=["Load Weight"])


# 2. CREATE DAILY WASTE DATA

daily_waste = (
    df.groupby("Report Date")["Load Weight"]
    .sum()
    .reset_index()
)

daily_waste.columns = ["Date", "Total_Waste"]

daily_waste = daily_waste.sort_values("Date").reset_index(drop=True)


# 3. FEATURE ENGINEERING

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


# 4. FEATURES AND TARGET

features = [
    "Day_of_Week",
    "Month",
    "Day",
    "Previous_Day_Waste",
    "Previous_7_Day_Avg"
]

X = daily_waste[features]
y = daily_waste["Total_Waste"]


# 5. CHRONOLOGICAL TRAIN-TEST SPLIT

split_index = int(len(daily_waste) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training period:",
    daily_waste["Date"].iloc[0].date(),
    "to",
    daily_waste["Date"].iloc[split_index - 1].date()
)

print(
    "Testing period:",
    daily_waste["Date"].iloc[split_index].date(),
    "to",
    daily_waste["Date"].iloc[-1].date()
)


# 6. CREATE MODELS

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}


# 7. TRAIN AND EVALUATE

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })


# 8. DISPLAY RESULTS

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("==============================")

print(
    results_df.to_string(index=False)
)