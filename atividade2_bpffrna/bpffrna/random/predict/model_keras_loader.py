import keras as k
import pandas as p
from sklearn.model_selection import train_test_split

datasetx = p.read_csv('../datset.csv', sep = ',', decimal = '.', usecols = ['x1', 'x2'])
datasety = p.read_csv('../datset.csv', sep = ',', decimal = '.', usecols = ['y'])
#def main_train():
#prepara dados
y = datasety['y']
x = datasetx[['x1', 'x2']]

x_treino, x_teste = train_test_split(x, test_size=0.2, random_state=42)
y_treino, y_teste = train_test_split(y, test_size=0.2, random_state=42)

model_files = ['keras_rna6.h5', 'keras_rna7.h5', 'keras_rna8.h5', 'keras_rna9.h5', 'keras_rna10.h5']

for model_file in model_files:
    model = k.models.load_model(model_file)

    loss, acc = model.evaluate(x_treino, y_treino)

    print(f"Model: {model_file}, TREINO Loss: {loss:.4f}, Accuracy: {acc:.2%}")
