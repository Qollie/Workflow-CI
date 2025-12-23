import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DataDf = pd.read_csv("Employee_preprocessing.csv")

FeatureDf = DataDf.drop("PaymentTier", axis=1)
TargetSeries = DataDf["PaymentTier"]

XTrain, XTest, YTrain, YTest = train_test_split(
    FeatureDf,
    TargetSeries,
    test_size=0.2,
    random_state=42
)

Model = RandomForestClassifier(random_state=42)
Model.fit(XTrain, YTrain)

Predictions = Model.predict(XTest)
Accuracy = accuracy_score(YTest, Predictions)

print(f"Training selesai. Accuracy: {Accuracy:.4f}")
