from tratamento_dados_data_set import carregar_dados;
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    X, y = carregar_dados()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size = 0.2,
    random_state=42)

    mu = Xtr.mean(axis=0);
    sd = Xtr.std(axis=0);

    Xtr = (Xtr - mu) / sd;
    Xte = (Xte - mu) / sd;

    print("Xtr padronizado: ", Xtr.shape, "| Xte padronizado: ", Xte.shape)