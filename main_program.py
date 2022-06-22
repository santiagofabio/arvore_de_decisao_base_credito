from copyreg import pickle
import pandas as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score,classification_report
import  matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix
import pickle


base_credit = np.read_csv('credit_data.csv')
print(f'{base_credit.head(3)}')
print(base_credit.describe())
print(f'{base_credit.columns }')

# Importando os atributos previsoes e classe ja tratados
with open('credit.pkl', 'rb') as f:
       x_credito_treinamento,y_credito_treinamento,x_credito_teste,y_credito_teste  = pickle.load(f)


print(f'Dimensões x_credito_treinamento:{x_credito_treinamento.shape}')
print(f'Dimensões y_credito_treinamento: {y_credito_treinamento.shape}')



print(f'Dimensões x_credito_teste:{x_credito_teste.shape}')
print(f'Dimensões y_credito_teste:{y_credito_teste.shape}')


arvore_credito = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credito.fit(x_credito_treinamento,y_credito_treinamento)
feature_importances =arvore_credito.feature_importances_
print(f'{feature_importances}')

previsoes = arvore_credito.predict(x_credito_teste)
accuracy  = accuracy_score(y_credito_teste, previsoes)
print(f' Precisão {accuracy}')

cm = ConfusionMatrix(arvore_credito)
cm.fit( x_credito_treinamento, x_credito_treinamento, rownames =['Real'], colnames =['predicto'], margins = True)
score_cm = cm.score(x_credito_teste,y_credito_teste)
print(f'Score Confusion Matriz {score_cm}')
plt.savefig("matriz_de_confusao.png", dpi =300, format='png') 
cm.show()


print(f'Parametros  construcao da arvore {classification_report(y_credito_teste,previsoes)}')


previsores =['Renda', 'Idade', 'Divida' ]
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

tree.plot_tree(arvore_credito, feature_names=previsores, class_names=['0','1'] , filled= True)
plt.savefig("arvore_de_decisao.png", dpi =300, format='png')
plt.show()