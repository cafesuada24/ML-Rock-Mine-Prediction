from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

random_state = 2024


def train(dataset_path: Path, output_path: Path):
    df = pd.read_csv(dataset_path, header=None)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.1, random_state=random_state, stratify=y
    )

    penalty = 'l2'
    C = 1.0
    class_weight = 'balanced'
    solver = 'liblinear'

    model = LogisticRegression(penalty=penalty, C=C, class_weight=class_weight, solver=solver)


    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(
        f"""
        TEST RESULT
        Accuracy: {accuracy}
    """
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        dump(model, f)
