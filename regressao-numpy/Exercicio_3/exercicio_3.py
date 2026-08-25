import numpy as np

from data.tratamento_dados_data_set import carregar_dados
from Exercicio_1.exercicio_1 import (
    dividir_treino_teste,
    padronizar,
    adicionar_intercepto,
    equacao_normal,
    prever,
)

from Exercicio_2.exercicio_2 import mse, rmse

def main():
    # Recriando o pipeline dos exercícios anteriores
    X, y = carregar_dados()
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)
    X_train_std, X_test_std, mu, sigma = padronizar(X_train, X_test)

    Xtr = adicionar_intercepto(X_train_std)
    Xte = adicionar_intercepto(X_test_std)

    _, w, _ = equacao_normal(Xtr, y_train)

    # RMSE do teste (erro "típico" do modelo, calculado no Exercício 2)
    y_pred_test = prever(Xte, w)
    rmse_test = rmse(y_test, y_pred_test)

    # TAREFA 1: Previsão concreta (1° linha do teste)

    # Pega a primeira amostra do teste, ainda na escala ORIGINAL
    x_exemplo = X_test[0] # Features cruas (não padronizadas ainda)
    y_real = y_test[0]    # Valor real de mpg dessa amostra

    # Padroniza usando mu e sigma DO TREINO
    x_exemplo_std = (x_exemplo - mu) / sigma

    # Adiciona o "1" na frente, pra multiplicar certo com o intercepto w0
    x_exemplo_final = np.hstack([1, x_exemplo_std])

    # Previsão: multiplica pesos pelas features (produto escalar)
    y_previsto = x_exemplo_final @ w

    print("--- Tarefa 1 e 2: Previsão concreta ---\n")
    print(f"Valor real (mpg): {y_real:.2f}")
    print(f"Valor previsto (mpg): {y_previsto:.2f}")

    # Tarefa 2: Erro absoluto dessa amostra
    erro_absoluto = abs(y_real - y_previsto)
    print(f"Erro absoluto: {erro_absoluto:.2f} mpg\n")

    # Tarefa 3: comparar com o erro típico (RMSE do teste)
    print("--- Tarefa 3: Comparando com o erro típico ---\n")
    print(f"RMSE do teste (Erro típico): {rmse_test:.2f} mpg")
    if erro_absoluto > rmse_test:
        print(f"Essa amostra errou MAIS que o normal "
              f"(errou {erro_absoluto:.2f}, típico é {rmse_test:.2f}).\n")
    else:
        print(f"Essa amostra errou MENOS que o normal "
              f"(errou {erro_absoluto:.2f}, típico é {rmse_test:.2f}).\n")

    # Tarefa 4 (desafio): feature mais influente
    feature_names = ["cylinders", "displacement", "horsepower",
                     "weight", "acceleration", "model_year"]

    # Pega o peso de maior módulo, ignorando o intercepto (w[0])
    idx_max = np.argmax(np.abs(w[1:])) + 1
    nome_feature_max = feature_names[idx_max - 1]
    peso_max = w[idx_max]

    print("--- Tarefa 4: Feature mais influente ---")
    print(f"Feature de maior peso: {nome_feature_max} (peso = {peso_max:.3f})")

    # Cria uma amostra "na média" (todas as features padronizadas = 0)
    x_media_std = np.zeros(len(feature_names))
    x_media_final = np.hstack([1, x_media_std])
    previsao_media = x_media_final @ w

    # Agora aumenta SÓ a feature mais influente em +1 desvio padrão
    x_alterado_std = x_media_std.copy()
    x_alterado_std[idx_max - 1] = 1.0
    x_alterado_final = np.hstack([1, x_alterado_std])
    previsao_alterada = x_alterado_final @ w

    mudanca = previsao_alterada - previsao_media

    print(f"Previsão com todas as features na média: {previsao_media:.2f} mpg")
    print(f"Previsão aumentando '{nome_feature_max}' em +1 desvio padrão: "
          f"{previsao_alterada:.2f} mpg")

    print(f"Mudança no mpg previsto: {mudanca:.2f} mpg")
    print(f"Faz sentido: o peso dessa feature era {peso_max:.3f}" 
          f"e a mudança bateu exatamente com esse valor.")
        
    
if __name__ == "__main__":
    main()