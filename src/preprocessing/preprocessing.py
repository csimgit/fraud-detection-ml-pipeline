from sklearn.preprocessing import StandardScaler


def preprocess(df):
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X["Amount_scaled"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))
    X = X.drop("Amount", axis=1)

    return X, y
