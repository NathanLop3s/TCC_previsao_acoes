from sklearn.neural_network import MLPRegressor


def criar_mlp(hidden_layer_sizes: tuple, seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
        random_state=seed,
    )