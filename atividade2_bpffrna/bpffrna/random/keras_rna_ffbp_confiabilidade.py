import keras as k
import tensorflow as tf
#from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import pandas as p
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ann_visualizer.visualize import ann_viz
import graphviz
warnings.filterwarnings('ignore')

model = k.models.Sequential()
datasetx = p.read_csv('datset_rep.csv', sep = ',', decimal = '.', usecols = ['x1', 'x2'])
datasety = p.read_csv('datset_rep.csv', sep = ',', decimal = '.', usecols = ['y'])
csv_logger = CSVLogger('epoch_logs.csv')

#def main_train():
#prepara dados
y = datasety['y']
x = datasetx[['x1', 'x2']]

x_treino, x_teste = train_test_split(x, test_size=0.2, random_state=42)
y_treino, y_teste = train_test_split(y, test_size=0.2, random_state=42)

#tensorboard_callback = TensorBoard(log_dir='./')
csv_logger = CSVLogger('epoch_logs.csv')
checkpoints = ModelCheckpoint('chechpoint.hdf5', monitor='val_loss', verbose = 1, save_best_only = True, mode = 'max')
early_stop = EarlyStopping(monitor = 'val_loss', patience = 50, verbose = 1)
callback_list = [csv_logger, checkpoints]
y_treino = y_treino / 10
y_teste  = y_teste  / 10
#configura rede
model.add(k.layers.Dense(units = 5, input_dim = x_treino.shape[1]))
model.add(k.layers.Dense(units = 2, activation = 'relu'))
model.add(k.layers.Dense(units = 5, activation = 'relu'))
model.add(k.layers.Dense(units = 2, activation = 'relu'))
model.add(k.layers.Dense(units = 1, activation = 'linear'))

#treina rede
model.compile(loss = 'mse', optimizer = 'sgd', metrics = ['mae'])
#k.utils.plot_model(model, show_shapes = True, expand_nested = True, to_file='./logs/viz/keras_representation.png')
#ann_viz(model, './logs/viz/keras_rna.gv', title='Keras RNA')
r = model.fit(x_treino, y_treino, epochs = 300, batch_size = 100, validation_data = (x_teste, y_teste), callbacks=[callback_list])
treino_loss, treino_accuracy = model.evaluate(x_treino, y_treino)
teste_loss, teste_accuracy = model.evaluate(x_treino, y_treino)
print(f" TREINO Loss: {treino_loss:.4f}, Accuracy: {treino_accuracy:.2%}")
print(f" TESTE Loss: {teste_loss:.4f}, Accuracy: {teste_accuracy:.2%}")
model.save('rna1.h5')
with open('weights.txt', 'w') as file:
    for lnum, layer in enumerate(model.layers):
        weigths = layer.get_weights()[0]
        biases  = layer.get_weights()[1]

        for nnum, bias in enumerate(biases):
            file.write(f' Camada {lnum}: Neurônio {nnum}: Bias =========> {bias} \n')
        
        for nnum, wgt in enumerate(weigths):
            file.write(f' Camada {lnum}: Neurônio {nnum}: Weigths ======> \n {wgt} \n')

        file.write('\n\n')

        #file.write(weigths)
        #file.write(biases)

#with open('weights.txt', 'w') as file:
#    for item in model.get_weights():
#        file.write(str(item))

#main_train()
#pred = model.predict(x_treino)
#print(pred)
#exibição dataset
raw_data = plt
raw_data.scatter(y_treino, x_treino['x1'], c = 'g', label='TREINO - Alimentador 1')
raw_data.scatter(y_treino, x_treino['x2'], c = 'r', label='TREINO - Alimentador 2')
raw_data.scatter(y_teste , x_teste['x1'] , c = 'm', label='TESTE  - Alimentador 1')
raw_data.scatter(y_teste , x_teste['x2'] , c = 'k', label='TESTE  - Alimentador 2')
raw_data.title('Dataset - Dispersão de dados')
raw_data.ylabel('Tensão de saída')
raw_data.xlabel('Posição')
raw_data.legend(['X1 treino', 'X2 treino', 'X1 teste', 'X2 teste'])
raw_data.show()

#exibição de resultados da rede neural
rna = plt
rna.plot(r.history['loss'])
rna.plot(r.history['val_loss'])
rna.plot(r.history['mae'])
rna.plot(r.history['val_mae'])
rna.title('Histórico de treinamento')
rna.ylabel('Função de custo')
rna.xlabel('Epocas de treinamento')
rna.legend(['MSE treino', 'MSE teste', 'MAE treino', 'MAE teste'])
rna.show()