import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
data = np.genfromtxt(r'C:/Users/kayow/desktop/EMGsDataset.csv', delimiter=',')

# QUESTÃO 1 — organizar os dados em X e Y
# transpor e organizar
data_t = data.T
X = data_t[:, 0:2]              #colunas 0 e 1 = sensores
y = data_t[:, 2].astype(int)    #coluna 2 = classe

nomes_expressoes = {
    1: 'Neutro',
    2: 'Sorriso',
    3: 'Sobrancelha levantada',
    4: 'Surpreso',
    5: 'Rabugento'
}

cores_emocoes = {
    1: 'gray',
    2: 'gold',
    3: 'forestgreen',
    4: 'deepskyblue',
    5: 'crimson'
}
classes = np.unique(y)

lista_cores = [cores_emocoes[c] for c in classes]
cmap_fundo = ListedColormap(lista_cores)

# QUESTÃO 2 — visualização do gráfico
plt.figure(figsize=(8, 6))
for c in classes:
    mask = (y == c)
    plt.scatter(X[mask, 0], X[mask, 1],
                color=cores_emocoes[c],
                label=nomes_expressoes[c],
                edgecolors='black',
                linewidth=0.3,
                s=10,
                alpha=0.6)

plt.title('Gráfico de Espalhamento — EMG por Expressão Facial')
plt.xlabel('Sensor 1 — Corrugador do Supercílio (μV)')
plt.ylabel('Sensor 2 — Zigomático Maior (μV)')
plt.legend(title='Expressões Faciais')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
#plt.show()

# QUESTÃO 3 — modelos de classificação
# calculos estatísticos base
means   = {c: X[y == c].mean(axis=0) for c in classes}
covs    = {c: np.cov(X[y == c].T) for c in classes}
prioris = {c: len(X[y == c]) / len(X) for c in classes}

cov_global = np.cov(X.T)
cov_pool   = sum((len(X[y == c]) / len(X)) * covs[c] for c in classes)

def gaussian_pdf_vetorizado(X_data, mu, sigma):
    p = mu.shape[0]

    # garantir q sigma seja inversível
    sigma = sigma + np.eye(p) * 1e-6
    sign, log_det = np.linalg.slogdet(sigma)
    inv = np.linalg.inv(sigma)

    diff = X_data - mu

    mahal = np.einsum('ij,jk,ik->i', diff, inv, diff)

    log_prob = -0.5 * (p * np.log(2 * np.pi) + log_det + mahal)
    return log_prob

def treinar_mqo(X_tr, y_tr):
    # treina MQO e retorna os coeficientes beta
    # compara vetor y_tr com vetor de classes
    Y_oh = (y_tr[:, None] == classes[None, :]).astype(float)
    X_b  = np.c_[np.ones(X_tr.shape[0]), X_tr]
    beta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ Y_oh
    return beta

def predizer_mqo(X_data, beta):
    X_b = np.c_[np.ones(X_data.shape[0]), X_data]
    return classes[np.argmax(X_b @ beta, axis=1)]

def treinar_gaussiano(X_tr, y_tr):
    m   = {c: X_tr[y_tr == c].mean(axis=0) for c in classes}
    cvs = {c: np.cov(X_tr[y_tr == c].T)    for c in classes}
    pri = {c: len(X_tr[y_tr == c]) / len(X_tr) for c in classes}
    c_pool   = sum((len(X_tr[y_tr == c]) / len(X_tr)) * cvs[c] for c in classes)
    c_global = np.cov(X_tr.T)
    return m, cvs, pri, c_pool, c_global

def predizer_gaussiano(X_data, m, cvs, pri, c_pool, c_global, model_type='qda', lam=0.5):
    log_scores = np.zeros((X_data.shape[0], len(classes)))

    for i, c in enumerate(classes):
        # seleciona a covariância correta para o modelo
        if   model_type == 'qda':      s = cvs[c]
        elif model_type == 'lda':      s = c_pool
        elif model_type == 'agregada': s = c_global
        elif model_type == 'naive':    s = np.diag(np.diag(cvs[c]))
        elif model_type == 'friedman': s = (1 - lam) * cvs[c] + lam * c_pool

        log_scores[:, i] = gaussian_pdf_vetorizado(X_data, m[c], s) + np.log(pri[c])

    # retorna a classe de maior score para cada amostra
    return classes[np.argmax(log_scores, axis=1)]

# treino com todos os dados
beta_global = treinar_mqo(X, y)
m_g, cvs_g, pri_g, cpool_g, cglobal_g = treinar_gaussiano(X, y)

print("=" * 50)
print("Acurácia por Modelo (dados completos):")
print("=" * 50)

acc_mqo = np.mean(predizer_mqo(X, beta_global) == y)
print(f"- MQO Tradicional:                  {acc_mqo:.2%}")

for nome, tipo in [
    ("Gaussiano Tradicional (QDA)",   'qda'),
    ("Covariâncias Iguais (LDA)",     'lda'),
    ("Matriz Agregada",               'agregada'),
    ("Naive Bayes",                   'naive'),
    ("Friedman (λ=0.5)",              'friedman'),
]:
    acc = np.mean(predizer_gaussiano(X, m_g, cvs_g, pri_g, cpool_g, cglobal_g, tipo) == y)
    print(f"- {nome}: {acc:.2%}")

# zonas de decisão
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

print("\nCalculando as zonas de decisão")

plot_models = {
    "MQO":                            None,
    "Gaussiano Tradicional (QDA)":    'qda',
    "Covariâncias Iguais (LDA)":      'lda',
    "Matriz Agregada":                'agregada',
    "Naive Bayes":                    'naive',
    "Friedman (Regularizado)":        'friedman'
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, (title, model_type) in enumerate(plot_models.items()):
    ax = axes[idx]

    if title == "MQO":
        Z = predizer_mqo(grid_points, beta_global)
    else:
        Z = predizer_gaussiano(grid_points, m_g, cvs_g, pri_g, cpool_g, cglobal_g, model_type)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_fundo)

    for c in classes:
        mask = (y == c)
        ax.scatter(X[mask, 0], X[mask, 1],
                   color=cores_emocoes[c],
                   label=nomes_expressoes[c] if idx == 0 else "",
                   edgecolors='black',
                   linewidth=0.5,
                   s=20,
                   alpha=0.9)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Sensor 1 (μV)')
    ax.set_ylabel('Sensor 2 (μV)')
    ax.grid(True, linestyle='--', alpha=0.3)

fig.legend(title="Expressões Faciais",
           loc='upper center',
           bbox_to_anchor=(0.5, 1.05),
           ncol=5,
           frameon=True)
plt.tight_layout()
#plt.show()

# QUESTÃO 4 — K-Fold Cross Validation p/ encontrar o lambda ideal
lambdas = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
K = 5
print("\n" + "=" * 50)
print(f"Questão 4 — K-Fold Cross Validation (K={K}) para λ ideal do Friedman:")
print("=" * 50)

acc_por_lambda = []

for lam in lambdas:
    N = len(X)
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, K)

    accs_fold = []
    for k in range(K):
        # indices de validação e treino
        val_idx   = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # treinar, avaliar  com este lambda
        m_k, cvs_k, pri_k, cpool_k, cglobal_k = treinar_gaussiano(X_tr, y_tr)
        y_hat = predizer_gaussiano(X_val, m_k, cvs_k, pri_k, cpool_k, cglobal_k,
                                   model_type='friedman', lam=lam)
        accs_fold.append(np.mean(y_hat == y_val))

    acc_media = np.mean(accs_fold)
    acc_por_lambda.append(acc_media)
    print(f"  λ = {lam:.3f}  →  Acurácia média: {acc_media:.4%}")

lambda_ideal = lambdas[np.argmax(acc_por_lambda)]
print(f"\n  >>> λ ideal encontrado: {lambda_ideal}")

# plotar curva de acurácia por lambda
plt.figure(figsize=(8, 4))
plt.plot(lambdas, acc_por_lambda, marker='o', color='steelblue')
plt.axvline(lambda_ideal, color='crimson', linestyle='--', label=f'λ ideal = {lambda_ideal}')
plt.title(f'Cross Validation — Acurácia do Friedman por λ (K={K})')
plt.xlabel('λ')
plt.ylabel('Acurácia Média')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# QUESTÃO 5 — validação p/ Monte Carlo (R=500, 80% treino/20% teste)
print("\n" + "=" * 50)
print("Questão 5 — Monte Carlo (R=500 rodadas, 80/20 split):")
print("=" * 50)

R = 500
N = len(X)
n_treino = int(0.8 * N)

# listas para acumular acurácias de cada modelo
hist_mqo      = []
hist_qda      = []
hist_lda      = []
hist_agregada = []
hist_naive    = []
hist_friedman = []

for rodada in range(R):
    # 80/20
    idx = np.random.permutation(N)
    idx_tr  = idx[:n_treino]
    idx_te  = idx[n_treino:]

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]

    #MQO
    beta_r = treinar_mqo(X_tr, y_tr)
    hist_mqo.append(np.mean(predizer_mqo(X_te, beta_r) == y_te))

    #gaussianos
    m_r, cvs_r, pri_r, cpool_r, cglobal_r = treinar_gaussiano(X_tr, y_tr)

    hist_qda.append(np.mean(
        predizer_gaussiano(X_te, m_r, cvs_r, pri_r, cpool_r, cglobal_r, 'qda') == y_te))
    hist_lda.append(np.mean(
        predizer_gaussiano(X_te, m_r, cvs_r, pri_r, cpool_r, cglobal_r, 'lda') == y_te))
    hist_agregada.append(np.mean(
        predizer_gaussiano(X_te, m_r, cvs_r, pri_r, cpool_r, cglobal_r, 'agregada') == y_te))
    hist_naive.append(np.mean(
        predizer_gaussiano(X_te, m_r, cvs_r, pri_r, cpool_r, cglobal_r, 'naive') == y_te))
    hist_friedman.append(np.mean(
        predizer_gaussiano(X_te, m_r, cvs_r, pri_r, cpool_r, cglobal_r, 'friedman',
                           lam=lambda_ideal) == y_te))

    if (rodada + 1) % 100 == 0:
        print(f"  Rodada {rodada + 1}/{R} concluída...")

# QUESTÃO 6 - Tabela
resultados = {
    "MQO Tradicional":                     hist_mqo,
    "Classificador Gaussiano (QDA)":       hist_qda,
    "Gaussiano (Cov. todo cj. treino)":    hist_lda,
    "Gaussiano (Cov. Agregada)":           hist_agregada,
    "Naive Bayes":                         hist_naive,
    f"Friedman (λ={lambda_ideal})":        hist_friedman,
}

sep = "-" * 80
print("\n" + "=" * 80)
print("Questão 6 — Resultados Monte Carlo (R=500 rodadas)")
print("=" * 80)
print(f"{'Modelos':<42} {'Média':>8} {'Desvio-P.':>10} {'Maior':>8} {'Menor':>8}")
print(sep)

for nome, hist in resultados.items():
    h = np.array(hist)
    print(f"{nome:<42} {h.mean():>8.4f} {h.std():>10.4f} {h.max():>8.4f} {h.min():>8.4f}")

print("=" * 80)

#gráfico histograma dos modelos
cores_modelos = ['steelblue', 'forestgreen', 'crimson', 'darkorange', 'mediumpurple', 'goldenrod']

plt.figure(figsize=(10, 5))
for (nome, hist), cor in zip(resultados.items(), cores_modelos):
    plt.hist(hist, bins=30, alpha=0.5, color=cor, label=nome, edgecolor='none')

plt.title('Distribuição das Acurácias — Monte Carlo (R=500)')
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()