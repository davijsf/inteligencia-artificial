import numpy as np
from sklearn.model_selection import train_test_split
# 1. Carregar e inspecionar
from data.tratamento_dados_data_set import carregar_dados

# 2. Split
def dividir_treino_teste(X, y, test_size=0.2, random_state=42):
    """Tarefa 2: Split 80/20 reprodutível."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 3. Padronização
# Coloca todas as features na mesma escala 
def padronizar(X_train, X_test):
    """Tarefa 3: Padronização z-score usando média/desvio do treino"""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    return X_train_std, X_test_std, mu, sigma

# Matriz de design
def adicionar_intercepto(X):
    """Tarefa 4: Adiciona a coluna de 1s para o termo w0."""
    m = X.shape[0]
    return np.hstack([np.ones((m,1)), X])


# 5. Equação normal
def equacao_normal(Xtr, y_train):
    """Tarefa 5: w* = (XtX)^-1 Xty, comparando inv() e solve()."""
    
    XtX = Xtr.T @ Xtr # Transposta de X * X
    Xty = Xtr.T @ y_train # Transposta de X * y

    w_inv = np.linalg.inv(XtX) @ Xty # calcula w = (XtX)^-1 * Xty de forma direta
    w_solve = np.linalg.solve(XtX, Xty) # resolve o sistema linear: XtXw = Xty sem inverter a mariz

    diff = np.max(np.abs(w_inv - w_solve)) # compara os resultados
    return w_inv, w_solve, diff

# 6. Previsões
def prever(X, w):
    """Tarefa 6: previsão yhat = Xw"""
    return X @ w

# 7. Interpretação dos pesos
def interpretar_pesos(w, feature_names):
    """Tarefa 7: identifica a feature de maior peso em módulo"""
    idx_max = np.argmax(np.abs(w[1:])) + 1 # ignora o intercepto
    return feature_names[idx_max], w[idx_max]

def main():
    # 1. Carregar e inspecionar
    X, y = carregar_dados()
    print("X: ", X.shape, "|y: ", y.shape)
    print(f"Sobraram {X.shape[0]} amostras após remover valores faltantes "
        f"(de 398 originais, foram descartadas {398 - X.shape[0]} linhas com '?').\n")
    

    # 2. Split
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)
    print("Treino: ", X_train.shape, "| Teste:", X_test.shape, "\n")

    # 3. Padronização
    X_train_std, X_test_std, mu, sigma = padronizar(X_train, X_test)

    # 4. Matriz de design (coluna de 1s)
    Xtr = adicionar_intercepto(X_train_std)
    Xte = adicionar_intercepto(X_test_std)
    print("Xtr: ", Xtr.shape, "| Xte: ", Xte.shape, "\n")

    # 5. Equação normal
    w_inv, w_solve, diff = equacao_normal(Xtr, y_train)
    print("w (inv): ", w_inv)
    print("w (solve): ", w_solve)
    print("Diferença máxima entre os métodos: ", diff, "\n")

    w = w_solve # usamos a versão mais estável numericamente

    # 6. Previsões
    y_pred_train = prever(Xtr, w)
    y_pred_test = prever(Xte, w)

    print("Previsão vs Real (5 primeiros pontos do teste): ")
    for i in range(5):
        print(f" Previsto: {y_pred_test[i]:.2f} | Real: {y_test[i]:.2f}")
    print()

    # 7. Interpretação dos pesos
    feature_names = ["intercepto", "cylinders", "displacement",
                     "horsepower", "weight", "acceleration"]

    for name, peso in zip(feature_names, w):
        print(f"{name:.12s}: {peso:.3f}")

    nome_max, peso_max = interpretar_pesos(w, feature_names)
    print(f"\nFeature de maior peso (em módulo): {nome_max} (w = {peso_max:.3f})")

if __name__ == "__main__":
    main()