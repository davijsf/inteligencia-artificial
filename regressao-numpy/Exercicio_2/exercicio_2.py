import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from data.tratamento_dados_data_set import carregar_dados
from Exercicio_1.exercicio_1 import (
    dividir_treino_teste,
    padronizar,
    adicionar_intercepto,
    equacao_normal,
    prever,
)

# MSE - mede o erro médio do modelo (erro^2)
def mse(y_real, y_pred):
    """Tarefa 1: MSE = (1/n) * soma((y-yhat)^2), com numpy."""
    erros = y_real - y_pred
    return np.mean(erros ** 2)

# Raiz quadrada do MSE. Desfaz o quadrado e volta
# à unidade original (mpg)
def rmse(y_real, y_pred):
    """Tarefa 2: RMSE = raiz(MSE)."""
    return np.sqrt(mse(y_real, y_pred))

# Repetindo em PyTorch
# Faz a mesma conta (previsao, MSe, RMSE), usando tensores do pytorch
def mse_rmse_torch(X_tensor, y_tensor, w_tensor):
    """Tarefa 3. previsão e MSE/RMSE usando só operações de tensor."""
    y_pred = X_tensor @ w_tensor
    erros = y_tensor - y_pred
    mse_t = (erros ** 2).mean()
    rmse_t = torch.sqrt(mse_t)
    return y_pred, mse_t, rmse_t

def main():
    # Reaproveitando o pipeline do EXercício 1
    X, y = carregar_dados()
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)
    X_train_std, X_test_std, mu, sigma = padronizar(X_train, X_test)

    Xtr = adicionar_intercepto(X_train_std)
    Xte = adicionar_intercepto(X_test_std)

    w_inv, w_solve, diff = equacao_normal(Xtr, y_train)
    w = w_solve

    y_pred_train = prever(Xtr, w)
    y_pred_test = prever(Xte, w)

    # Tarefas 1 e 2: MSE e RMSE com numpy
    mse_train = mse(y_train, y_pred_train)
    rmse_train = rmse(y_train, y_pred_train)
    mse_test = mse(y_test, y_pred_test)
    rmse_test = rmse(y_test, y_pred_test)

    print("--- MSE / RMSE ---")
    print(f"{'Conjunto':10s} {'MSE':>10s} {'RMSE':>10s}")
    print(f"{'Treino':10s} {mse_train:10.4f} {rmse_train:10.4f}")
    print(f"{'Teste':10s} {mse_test:10.4f} {rmse_test:10.4f}\n")

    # Tarefa 3: mesmas contas com pytorch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xte_tensor = torch.tensor(Xte, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
    w_tensor = torch.tensor(w, dtype=torch.float32, device=device)

    y_pred_test_torch, mse_test_torch, rmse_test_torch = mse_rmse_torch(
        Xte_tensor, y_test_tensor, w_tensor
    )

    print("--- MSE / RMSE (torch, device): ", device, ") ---")
    print(f"MSE (torch): {mse_test_torch.item():.4f}")
    print(f"RMSE (torch): {rmse_test_torch.item():.4f}\n")

    # Tarefa 4: comparação numpy vs torch
    # confirma que as duas dão praticamente o mesmo número
    diff_mse = abs(mse_test - mse_test_torch.item())
    diff_rmse = abs(rmse_test - rmse_test_torch.item())
    print("-- Diferença numpy vs torch --")
    print(f"Diferença MSE: {diff_mse:.2e}")
    print(f"Diferença RMSE: {diff_rmse:.2e}\n")

    # Tarefa 5: conferência com sklearn 
    # usa LinearRegression(uma biblioteca pronta para chevar se o calculo manual deu certo)
    modelo_sklearn = LinearRegression()
    modelo_sklearn.fit(X_train_std, y_train)
    y_pred_test_sklearn = modelo_sklearn.predict(X_test_std)

    mse_sklearn = mse(y_test, y_pred_test_sklearn)
    rmse_sklearn = rmse(y_test, y_pred_test_sklearn)

    print("--- Conferência com sklearn ---")
    print(f"MSE (manual): {mse_test:.4f} | MSE (sklearn): {mse_sklearn}")
    print(f"RMSE (manual): {rmse_test:.4f} | RMSE (sklearn): {rmse_sklearn}\n")
    print(f"Diferença MSE: {abs(mse_test - mse_sklearn):.2e}")
    print(f"Diferença RMSE: {abs(rmse_test - rmse_sklearn):.2e}\n")

    # Tarefa 7: Versão sem padronização
    # Refaz tudo sem colocar as features na mesma escala, para ver que o erro final nao muda muito
    # mas os pesos ficam dificeis e baguncados de comparar entre si
    Xtr_sem_pad = adicionar_intercepto(X_train)
    Xte_sem_pad = adicionar_intercepto(X_test)

    _, w_sem_pad, _ = equacao_normal(Xtr_sem_pad, y_train)

    y_pred_train_sem_pad = prever(Xtr_sem_pad, w_sem_pad)
    y_pred_test_sem_pad = prever(Xte_sem_pad, w_sem_pad)

    mse_train_sp = mse(y_train, y_pred_train_sem_pad)
    rmse_train_sp = rmse(y_train, y_pred_train_sem_pad)
    mse_test_sp = mse(y_test, y_pred_test_sem_pad)
    rmse_test_sp = rmse(y_test, y_pred_test_sem_pad)

    print("--- Sem padronização --- ")
    print(f"{'Conjunto':10s} {'MSE':>10s} {'RMSE:':>10s}")
    print(f"{'Treino':10s} {mse_train_sp:10.4f} {rmse_train_sp:10.4f}")
    print(f"{'Teste':10s} {mse_test_sp:10.4f} {rmse_test_sp:10.4f}\n")

    print("Pesos COM padronização:", w)
    print("\nPesos SEM padronização:", w_sem_pad)


if __name__ == "__main__":
    main()
