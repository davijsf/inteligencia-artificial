import pandas as pd;
from pathlib import Path

# Sobe um nível a partir de data/ até a raiz do projeto, depois entra em database/
CAMINHO_DADOS = Path(__file__).resolve().parent.parent / "database" / "auto-mpg.data"


def carregar_dados():
    # nomes das colunas
    cols = ["mpg", "cylinders", "displacement", "horsepower", 
            "weight", "acceleration", "model_year", "origin", "car_name"]

    df = pd.read_csv(CAMINHO_DADOS, sep=r"\s+", header=None, names=cols);

    # Remove as linhas "?":
    have_missing = df.astype(str).apply(
        lambda c: c.str.contains(r"\?", na=False)).any(axis=1)
    df = df[~have_missing]

    # Features (X) e alvo (y)
    # Removendo mpg (alvo), origin(categórica) e car_name (texto)
    X = df.drop(columns=["mpg", "origin", "car_name"]).to_numpy(dtype=float)
    y = df["mpg"].to_numpy(dtype=float)
    return X, y

if __name__ == "__main__":
    X, y = carregar_dados()
    print("X: ", X.shape, "|y: ", y.shape)