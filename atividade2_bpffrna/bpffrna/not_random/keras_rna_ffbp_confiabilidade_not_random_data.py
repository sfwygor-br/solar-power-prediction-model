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
x_treino = p.read_csv('base_treino_rep.csv', sep = ',', decimal = '.', usecols = ['x1', 'x2'])
y_treino = p.read_csv('base_treino_rep.csv', sep = ',', decimal = '.', usecols = ['y'])

x_teste = p.read_csv('base_teste_rep.csv', sep = ',', decimal = '.', usecols = ['x1', 'x2'])
y_teste = p.read_csv('base_teste_rep.csv', sep = ',', decimal = '.', usecols = ['y'])

#tensorboard_callback = TensorBoard(log_dir='./')
csv_logger = CSVLogger('epoch_logs.csv')
checkpoints = ModelCheckpoint('chechpoint.hdf5', monitor='val_loss', verbose = 1, save_best_only = True, mode = 'max')
early_stop = EarlyStopping(monitor = 'val_loss', patience = 15, verbose = 1)
callback_list = [csv_logger, checkpoints, early_stop]
y_treino = y_treino / 10
y_teste  = y_teste  / 10
#configura rede
model.add(k.layers.Dense(units = 2, input_dim = x_treino.shape[1]))
model.add(k.layers.Dense(units = 3, activation = 'softmax'))
model.add(k.layers.Dense(units = 3, activation = 'softmax'))
model.add(k.layers.Dense(units = 3, activation = 'softplus'))
model.add(k.layers.Dense(units = 3, activation = 'softplus'))
model.add(k.layers.Dense(units = 1))

#treina rede
model.compile(loss = 'mse', optimizer = 'sgd', metrics = ['mae'])
#k.utils.plot_model(model, show_shapes = True, expand_nested = True, to_file='./logs/viz/keras_representation.png')
#ann_viz(model, './logs/viz/keras_rna.gv', title='Keras RNA')
r = model.fit(x_treino, y_treino, epochs = 2000, batch_size = 100, validation_data = (x_teste, y_teste), callbacks=[callback_list])
treino_loss, treino_accuracy = model.evaluate(x_treino, y_treino)
teste_loss, teste_accuracy = model.evaluate(x_treino, y_treino)
print(f" TREINO Loss: {treino_loss:.4f}, Accuracy: {treino_accuracy:.2%}")
print(f" TESTE Loss: {teste_loss:.4f}, Accuracy: {teste_accuracy:.2%}")
model.save('keras_rna2.h5')
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
