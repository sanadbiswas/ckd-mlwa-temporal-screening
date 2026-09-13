def make_preprocessor(X):
    cat_cols = list(X.select_dtypes(exclude=np.number).columns)
    num_cols = list(X.select_dtypes(include=np.number).columns)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols)
    ])


def make_pipeline(model, X):
    return Pipeline([
        ("prep", make_preprocessor(X)),
        ("model", model)
    ])
