import keras as k
import pandas as p

x_treino = p.read_csv('../base_treino_rep.csv', sep=',', decimal='.', usecols=['x1', 'x2'])
y_treino = p.read_csv('../base_treino_rep.csv', sep=',', decimal='.', usecols=['y'])

x_teste = p.read_csv('../base_teste_rep.csv', sep=',', decimal='.', usecols=['x1', 'x2'])
y_teste = p.read_csv('../base_teste_rep.csv', sep=',', decimal='.', usecols=['y'])

y_treino = y_treino / 10
y_teste = y_teste / 10

model_files = ['keras_rna1.h5', 'keras_rna2.h5', 'keras_rna3.h5', 'keras_rna4.h5', 'keras_rna5.h5']

for model_file in model_files:
    model = k.models.load_model(model_file)

    loss, acc = model.evaluate(x_treino, y_treino)

    print(f"Model: {model_file}, TREINO Loss: {loss:.4f}, Accuracy: {acc:.2%}")
