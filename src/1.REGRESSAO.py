import numpy as np
import matplotlib.pyplot as plt

dados = np.loadtxt('C:/Users/kayow/desktop/aerogerador.dat')

# QUESTÃO 1 - gráfico
vento = dados[:, 0]
potencia = dados[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(vento, potencia, alpha=0.3, s=8, color='crimson', edgecolors='k')

ax.set_xlabel('Velocidade do Vento (m/s)', fontsize=13)
ax.set_ylabel('Potência Gerada (kW)', fontsize=13)
ax.set_title('Gráfico de Espalhamento – Aerogerador\nVelocidade do Vento vs Potência Gerada', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
#plt.show()

# QUESTÃO 2 - organização dos dados em matriz X e vetor y
N = dados.shape[0]

uns = np.ones((N, 1))
x_col = vento.reshape(N, 1)
X = np.hstack((uns, x_col))

y = potencia.reshape(N, 1)

# QUESTÃO 3 - implementação dos modelos
# 1: Média da variável dependente w = [média(y), 0]
w_media = np.array([[np.mean(y)], [0.0]])

# 2: MQO Tradicional w = (X^T*X)^(-1)*X^T*y
w_mqo = np.linalg.pinv(X.T @ X) @ X.T @ y

# 3: MQO Regularizado (Tikhonov) w = (X^T*X + lambda * I)^(-1)*X^T*y
lambdas = [0, 0.25, 0.5, 0.75, 1]

I = np.eye(X.shape[1])
I[0, 0] = 0

w_tikhonov = {}
for lam in lambdas:
    w_tikhonov[lam] = np.linalg.pinv(X.T @ X + lam * I) @ X.T @ y

print("QUESTÃO 3 – Coeficientes Estimado\n")
print(f"Média        → w0 (intercepto): {w_media[0,0]:.4f} | w1: {w_media[1,0]:.4f}")
print(f"MQO          → w0 (intercepto): {w_mqo[0,0]:.4f}  | w1: {w_mqo[1,0]:.4f}")
for lam in lambdas:
    w = w_tikhonov[lam]
    print(f"Tikhonov λ={lam} → w0 (intercepto): {w[0,0]:.4f}  | w1: {w[1,0]:.4f}")

x_linha = np.linspace(vento.min(), vento.max(), 200).reshape(-1, 1)
X_linha = np.hstack((np.ones((200, 1)), x_linha))

plt.figure(figsize=(10, 6))
plt.scatter(vento, potencia, color='forestgreen', alpha=0.2, s=6, label='Dados')

plt.axhline(y=w_media[0, 0], color='orange', linewidth=2, label='Média')
plt.plot(x_linha, X_linha @ w_mqo, 'r-', linewidth=2, label='MQO Tradicional')

cores = ['blue', 'green', 'purple', 'brown', 'cyan']
for lam, cor in zip(lambdas, cores):
    plt.plot(x_linha, X_linha @ w_tikhonov[lam], '--', color=cor,
             linewidth=1.5, label=f'Tikhonov λ={lam}')

plt.xlabel('Velocidade do Vento (m/s)', fontsize=13)
plt.ylabel('Potência Gerada (kW)', fontsize=13)
plt.title('Questão 3 – Ajuste dos Modelos', fontsize=14)
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()


# QUESTÃO 4 - validação dos lambdas [0, 0.25, 0.5, 0.75, 1]
R = 500
ponto_corte = int(0.8 * N)

resultados = {
    'media':  {'mse': [], 'r2': []},
    'mqo':    {'mse': [], 'r2': []},
}
for lam in lambdas:
    resultados[f'tik_{lam}'] = {'mse': [], 'r2': []}

def mse(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)

def r2(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - ss_res / ss_tot

# Questão 5: loop de R=500 rodadas
for _ in range(R):
    idx = np.random.permutation(N)
    X_tr, y_tr = X[idx[:ponto_corte]], y[idx[:ponto_corte]]
    X_te, y_te = X[idx[ponto_corte:]], y[idx[ponto_corte:]]

    # média
    y_pred = np.full_like(y_te, np.mean(y_tr))
    resultados['media']['mse'].append(mse(y_te, y_pred))
    resultados['media']['r2'].append(r2(y_te, y_pred))

    # MQO Tradicional
    w = np.linalg.pinv(X_tr.T @ X_tr) @ X_tr.T @ y_tr
    y_pred = X_te @ w
    resultados['mqo']['mse'].append(mse(y_te, y_pred))
    resultados['mqo']['r2'].append(r2(y_te, y_pred))

    # Tikhonov p/ cada lambda
    I_val = np.eye(X_tr.shape[1])
    I_val[0, 0] = 0
    for lam in lambdas:
        w = np.linalg.pinv(X_tr.T @ X_tr + lam * I_val) @ X_tr.T @ y_tr
        y_pred = X_te @ w
        resultados[f'tik_{lam}']['mse'].append(mse(y_te, y_pred))
        resultados[f'tik_{lam}']['r2'].append(r2(y_te, y_pred))

# Questão 6 - tabela
modelos_nomes = ['Média', 'MQO', 'Tik λ=0', 'Tik λ=0.25', 'Tik λ=0.5', 'Tik λ=0.75', 'Tik λ=1']
chaves        = ['media', 'mqo', 'tik_0', 'tik_0.25', 'tik_0.5', 'tik_0.75', 'tik_1']

print("\nTabela MSE")
print(f"{'Modelo':<20} {'Média':>10} {'DP':>10} {'Máx':>10} {'Mín':>10}")
for nome, chave in zip(modelos_nomes, chaves):
    m = np.array(resultados[chave]['mse'])
    print(f"{nome:<20} {np.mean(m):>10.2f} {np.std(m):>10.2f} {np.max(m):>10.2f} {np.min(m):>10.2f}")

print("\nTabela R²")
print(f"{'Modelo':<20} {'Média':>10} {'DP':>10} {'Máx':>10} {'Mín':>10}")
for nome, chave in zip(modelos_nomes, chaves):
    r2v = np.array(resultados[chave]['r2'])
    print(f"{nome:<20} {np.mean(r2v):>10.4f} {np.std(r2v):>10.4f} {np.max(r2v):>10.4f} {np.min(r2v):>10.4f}")   