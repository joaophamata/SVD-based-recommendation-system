import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.accuracy import rmse
from io import StringIO

# Seu conjunto de dados se tiver outros pode trocar

dados = """
Pessoas/Filmes,O Poderoso Chefão,O Senhor dos Anéis,Interestelar,Cidade de Deus,A Origem,Star Wars Ep. V,Pulp Fiction,Forrest Gump,Clube da Luta,Titanic,Matrix,Labirinto do Fauno,Silêncio dos Inocentes,O Grande Truque,Os Infiltrados,A Lista de Schindler,Vingadores: Ultimato,Os Sete Samurais,A Vida é Bela,Os Intocáveis,O Sexto Sentido,O Rei Leão,A Viagem de Chihiro,Gladiador,Crepúsculo dos Deuses,A Sociedade do Anel,Bastardos Inglórios,O Iluminado,Jurassic Park,La La Land,O Discurso do Rei,A Chegada,O Resgate do Soldado Ryan,O Segredo dos Seus Olhos,A Forma da Água,O Grande Lebowski,Os Suspeitos,A Bela e a Fera,Mad Max: Estrada da Fúria,O Show de Truman,E.T. - O Extraterrestre,O Fabuloso Destino de Amélie Poulain,Ponte Para Terabítia,O Casamento de Meu Melhor Amigo,Batman: O Cavaleiro das Trevas,Os Descendentes,O Jogo da Imitação,Os Caçadores da Arca Perdida,Moonlight,Casablanca,Superbad,A Escolha Perfeita,Meu Malvado Favorito,Thor: Ragnarok,Mulheres ao Ataque,Zumbilândia,A Ressaca,Amizade Colorida,A Entrevista,A Era do Gelo,Sing: Quem Canta Seus Males Espanta,Kung Fu Panda,Hotel Transilvânia,Up: Altas Aventuras,Megamente,Moana: Um Mar de Aventuras,O Touro Ferdinando,Rango,Divertida Mente,Shrek
Ação,75,100,75,75,75,100,100,75,100,25,100,75,25,25,75,25,100,75,25,75,25,75,75,75,100,75,100,75,100,25,75,75,100,25,75,75,75,25,100,25,75,25,25,25,100,75,75,100,75,75,75,25,75,100,25,100,75,25,75,75,75,100,75,75,100,75,75,100,75,75
Comediante,25,75,25,25,25,75,75,100,25,25,25,25,25,75,25,25,75,75,25,75,25,75,75,75,25,75,75,25,75,75,25,25,25,75,25,75,25,75,25,100,75,75,75,75,25,75,25,75,25,75,100,75,100,100,100,100,100,75,100,100,100,100,100,100,100,100,100,75,100,100
Romântica,75,75,75,25,75,75,25,75,25,100,25,75,75,75,75,100,25,75,75,100,75,75,100,75,75,75,75,25,75,100,75,75,75,75,100,75,25,100,25,75,75,100,75,100,25,100,75,75,75,100,25,75,25,25,75,25,25,75,25,25,75,25,75,75,75,75,25,75,75,75
Drama,100,100,100,100,100,75,100,75,100,75,100,100,100,75,75,100,25,100,100,100,100,75,75,75,75,100,100,100,75,75,100,100,100,100,100,75,100,75,75,75,75,75,75,75,75,25,100,75,100,75,25,75,25,25,25,25,25,25,25,25,75,25,25,75,25,25,25,75,25,25
Grandes Produções,100,100,75,75,100,100,75,75,25,100,100,100,75,75,75,100,100,75,75,75,75,100,75,100,75,100,75,75,100,75,75,75,75,75,75,75,75,75,100,75,75,25,75,75,100,75,75,100,75,75,75,75,75,100,75,75,75,75,75,75,75,75,75,100,75,100,75,75,100,75
"Cinema ""Arte""/Cult",100,100,100,100,100,100,100,100,100,75,100,100,100,75,100,100,25,100,100,100,100,75,100,75,100,100,100,100,100,100,100,100,100,100,100,75,100,100,100,100,75,25,100,25,75,100,100,100,25,25,25,25,25,25,75,25,25,25,25,25,25,25,25,25,25,75,25,25
"""

# Criar um DataFrame a partir dos dados
df = pd.read_csv(StringIO(dados))

n = 0.6 # Mudar pelo menos n% das avaliações para 50.0 se quiser mudar também
mask = np.random.rand(*df.iloc[:, 1:].shape) < n
df.iloc[:, 1:] = np.where(mask, 50.0, df.iloc[:, 1:])

# Imprimir a matriz original com n% de valores contínuos entre 1 e 100
print("Matriz Original com " + str(n) + "% de valores contínuos:")
print(df)

# Estruturar dados para o Surprise
reader = Reader(rating_scale=(1, 100))
data = Dataset.load_from_df(df.melt(id_vars='Pessoas/Filmes', var_name='Filme', value_name='Avaliacao'), reader)

# Dividir os dados em conjunto de treinamento e teste
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Definir os parâmetros a serem otimizados
param_grid = {'n_factors': [5, 10, 15, 20],
              'n_epochs': [10, 15, 20, 25],
              'lr_all': [0.002, 0.005, 0.008],
              'reg_all': [0.01, 0.02, 0.03]}

# Inicializar o modelo
svd = SVD()

# Inicializar o GridSearchCV
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)

# Executar o Grid Search no conjunto de treinamento
grid_search.fit(data)

# Obter os melhores parâmetros
best_params = grid_search.best_params['rmse']
n_factors = best_params['n_factors']
n_epochs = best_params['n_epochs']
lr_all = best_params['lr_all']
reg_all = best_params['reg_all']

# Inicializar o modelo com os melhores parâmetros
best_model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

# Treinar o modelo com os melhores parâmetros
best_model.fit(trainset)

# Fazer previsões no conjunto de teste
predictions = best_model.test(testset)

# Avaliar o desempenho do modelo otimizado
accuracy = rmse(predictions)
print(f'\nRMSE do modelo otimizado: {accuracy}')

# Geração de recomendações usando a função build_anti_testset()
testset_full = trainset.build_anti_testset()
all_predictions = best_model.test(testset_full)

# Estruturar as recomendações por usuário
recomendacoes_por_usuario = {}
for usuario_id, filme_id, _, nota_predita, _ in all_predictions:
    # Verificar se a nota original no DataFrame é igual a 50.0
    nota_original = df.loc[df['Pessoas/Filmes'] == usuario_id, filme_id].values[0]
    if nota_original == 50.0:
        if usuario_id not in recomendacoes_por_usuario:
            recomendacoes_por_usuario[usuario_id] = []
        recomendacoes_por_usuario[usuario_id].append((filme_id, nota_predita))

# Ordenar e obter as recomendações para cada usuário
for usuario_id, recomendacoes in recomendacoes_por_usuario.items():
    recomendacoes_por_usuario[usuario_id] = sorted(recomendacoes, key=lambda x: x[1], reverse=True)[:]

# Imprimir as recomendações para todos os usuários
print("\nRecomendações para todos os usuários:")
for usuario, recom in recomendacoes_por_usuario.items():
    print(f'{usuario}: {recom}')

from sklearn.decomposition import PCA

# Obter a matriz de fatores latentes para os usuários
fatores_latentes_usuarios = best_model.pu

# Reduzir a dimensionalidade com PCA para 2 componentes principais
pca = PCA(n_components=2)
usuarios_pca = pca.fit_transform(fatores_latentes_usuarios)

# Plotar os usuários no gráfico bidimensional
plt.figure(figsize=(10, 8))
plt.scatter(usuarios_pca[:, 0], usuarios_pca[:, 1])

# Rotular cada ponto com o ID do usuário
for i, usuario_id in enumerate(recomendacoes_por_usuario.keys()):
    plt.annotate(usuario_id, (usuarios_pca[i, 0], usuarios_pca[i, 1]))

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Representação dos Usuários nos Fatores Latentes (PCA)')
plt.show()

# Obter a matriz de fatores latentes para os filmes
fatores_latentes_filmes = best_model.qi

# Reduzir a dimensionalidade com PCA para 2 componentes principais
pca = PCA(n_components=2)
filmes_pca = pca.fit_transform(fatores_latentes_filmes)

# Plotar os filmes no gráfico bidimensional
plt.figure(figsize=(15, 15))
plt.scatter(filmes_pca[:, 0], filmes_pca[:, 1])

# Rotular cada ponto com o ID do filme
for i, filme_id in enumerate(df.columns[1:]):
    plt.annotate(filme_id, (filmes_pca[i, 0], filmes_pca[i, 1]))

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Representação dos Filmes nos Fatores Latentes (PCA)')
plt.show()

# Imprimir as recomendações para alguns usuários com todas as notas esperadas
num_usuarios_a_plotar = 6

for i, (usuario, recom) in enumerate(recomendacoes_por_usuario.items()):
    if i >= num_usuarios_a_plotar:
        break

    # Filmes recomendados e notas esperadas
    filmes_recomendados = [filme for filme, _ in recom]
    notas_preditas = [nota for _, nota in recom]

    # Todas as notas esperadas (inclusive para os não recomendados)
    todas_filmes = df.columns[1:]
    todas_notas_preditas = [best_model.predict(usuario, filme).est for filme in todas_filmes]

    # Plotagem
    plt.figure(figsize=(15, 15))
    plt.barh(todas_filmes, todas_notas_preditas, color='lightgray', label='Todas as Notas')
    plt.barh(filmes_recomendados, notas_preditas, color='skyblue', label='Recomendados')
    plt.xlabel('Nota Predita')
    plt.title(f'Notas Esperadas para o Usuário {usuario}')
    plt.legend()
    plt.show()
