from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURES = [
    "age",
    "height",
    "weight",
    "aids",
    "cirrhosis",
    "hepatic_failure",
    "immunosuppression",
    "leukemia",
    "lymphoma",
    "solid_tumor_with_metastasis",
]
TARGET = "diabetes_mellitus"


def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)


# updated function to accept feature_list
def train_model(train_df, feature_list=None):
    """Train logistic regression model."""
    X = train_df[feature_list or FEATURES]
    y = train_df[TARGET]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


# updated function to accept feature_list
def add_predictions(df, model, feature_list=None):
    """Add predicted probabilities to dataframe."""
    df["predictions"] = model.predict_proba(df[feature_list or FEATURES])[:, 1]
    return df
