# **Comparativo Final**

### 1. MSE e RMSE deram praticamente os mesmos valores entre numpy, tensores e sklearn? Qual a maior diferença?
Sim, os valores foram praticamente idênticos nos três métodos:
•  **numpy vs sklearn: ** MSE = 10.5024 em ambos; diferença de apenas 3.20 x 10-¹⁴ (essencialmente zero, erro de arredondamento de ponto flutuante)
•  **numpy vs torch: ** MSE = 10.5024 em ambos; diferença de 5.05 x 10⁻⁷.

A maior diferença encontrada foi entre numpy e torch (5.05 × 10⁻⁷), maior que a diferença numpy-sklearn (3.20 × 10⁻¹⁴). 
Isso é esperado: numpy e sklearn trabalham em float64 (dupla precisão) por padrão, enquanto o `PyTorch` usou float32 (precisão simples) 
na conversão dos tensores — float32 tem menos casas decimais de precisão, então acumula um pouco mais de erro de arredondamento nas operações. 
Ainda assim, ambas as diferenças são basicamente zero, para fins práticos.

### 2. O RMSE é maior ou menor que o MSE? Por quê?
O `RMSE` (3.2407) é menor que o MSE (10.5024) no teste. Isso acontece porque o MSE
é maior que 1, e a raiz quadrada de um número maior que 1 sempre resulta em um valor menor (√10.5024 ≈ 3.24).
Além do valor numérico, as unidades são diferentes: o `MSE` está em mpg² (milhas por galão ao quadrado) — 
uma unidade sem interpretação prática direta, resultado de elevar os erros ao quadrado. O RMSE, ao tirar a raiz, 
volta para a unidade original do problema, `mpg`, permitindo uma leitura direta: "o modelo erra, em média, cerca de 3.24 mpg" — 
muito mais interpretável que "10.5 mpg²".

### 3. Qual feature mais influencia o alvo? O sinal do peso faz sentido?
A feature que mais influencia o mpg é `weight` (peso = -5.510, o maior em módulo entre todas as features). 
O sinal negativo faz sentido físico: carros mais pesados exigem mais energia para se locomover, consumindo mais combustível e reduzindo o `mpg`.
Em segundo lugar, `model_year` (peso = +2.757) também é bastante influente, com sinal positivo — carros mais novos são mais eficientes, 
refletindo avanços tecnológicos em motores ao longo dos anos.
