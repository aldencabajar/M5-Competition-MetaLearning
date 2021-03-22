import tensorflow as tf
import numpy as np
import argparse
import pickle
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping 


def one_hot_fn(id_arr, ret_vocab = False):
  unique_ids = id_arr.unique()
  len_unique_ids = id_arr.value_counts().set_axis(unique_ids).values 
  stri_split = tf.strings.split(unique_ids, sep = "_")
  vocab, idx = tf.unique(stri_split.flat_values)
  encoded_ids = [one_hot(d, tf.size(vocab).numpy()) for d in unique_ids]
  max_length = np.max([len(i) for i in encoded_ids])
  padded_encoded = np.array(pad_sequences(encoded_ids, maxlen=max_length, padding='post'))
  padded_encoded = np.repeat(padded_encoded, len_unique_ids, axis = 0)

  if ret_vocab:
    return padded_encoded, vocab 
  else:
    return padded_encoded

train_id_enc, vocab = one_hot_fn(train_data['id'], ret_vocab = True)
validation_id_enc = one_hot_fn(validation_data['id'])
eval_id_enc = one_hot_fn(evaluation_data['id'])


def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return keras.backend.mean(v)



def create_model(feat_vars, n_hidden_layers, hidden_units, lr):
  
  ts_id_input = layers.Input(shape = (None, ), dtype = 'int32', 
                             name =  "ts_id_input") 
  # represent vocabulary with a 16-d vector                           
  embedding = layers.Embedding(836, 200) 
  encoded_input = embedding(ts_id_input)

  # since there are ids with multiple words, average vectors into single 
  # representation of the phrase
  pooled_input = layers.GlobalAveragePooling1D()(encoded_input)
  num_input = layers.Input(shape = (feat_vars.shape[0],) , dtype = 'float32', 
                              name = 'ts_features')
  concat_layer = layers.concatenate([pooled_input, num_input])

  # involve dropout layer for regularization 
  for i in range(n_hidden_layers):
    if i < (n_hidden_layers - 1) :
      if i == 0:
        mod = layers.Dense(hidden_units, name = 'layer' + str(i),
                          activation = 'relu')(concat_layer)
      else:
        mod = layers.Dense(hidden_units, name = 'layer' + str(i),
                          activation = 'relu')(mod)
      mod = layers.Dropout(0.3)(mod)
      mod = layers.Concatenate(name = "ft"+str(i))([pooled_input, mod])

    elif i == (n_hidden_layers - 1):
      mod = layers.Dense(hidden_units, name = 'layer' + str(i),
                         activation = 'relu')(mod)
  # output final dense layer with 9 outputs equal to the number of quantiles to 
  # be predicted                        
  preds = layers.Dense(9, activation="linear", name="preds")(mod)
  model = keras.Model(inputs = [ts_id_input, num_input], 
                      outputs = [preds])
  optim = keras.optimizers.Adam(learning_rate = lr)                   
  model.compile(loss=qloss, optimizer=optim)
  return(model)

  if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proc_data_path', type=str, help = 'path to processed data')
    parser.add_argument('-hu', '--num_hidden_units', type=int, default=200, help = 'number of hidden units')
    parser.add_argument('-hl', '--num_hidden_layers', type=int, default=3, help = 'number of hidden layers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help = 'learning rate')
    parser.add_argument('-f', '--features', nargs ="+", type=str, default=['day_of_year', 'year'], help = 'feature list')
    parser.add_argument('-w','--weights_file', type=str, help = 'path to save weights')

    args = parser.parse_args()
    feat_vars = ['id', 'sales','scaled', 'wt'] + args.features

    # create the model    
    model = create_model(feat_vars, n_hidden_layers = args.num_hidden_layers, 
    hidden_units = args.num_hidden_units, lr = args.learning_rate) 

    #setup other params

    ckpt = ModelCheckpoint(args.weights_file, 
    monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=2, min_lr=0.00001)
    es = EarlyStopping(monitor='val_loss', patience=5)
    keras.backend.clear_session()

    # load data dcits from proc_data_path
    data_dict_paths = os.listdir(args.proc_data_path)
    _dict_names = [s.replace('.pkl', '') for s in data_dict_paths]
    data_dict = {}

    for f, dict_name in zip(data_dict_paths, _dict_names):
      with open(f, 'rb') as _file:
        _dict = pickle.load(f)
        _dict['ts_features'] = _dict['ts_features'].loc[:, feat_vars]
        data_dict[dict_name] = _dict 

                    
    history = model.fit(x = data_dict['train_data'], y = train_data['scaled'].values, 
            validation_data = (data_dict['val_data'], validation_data['scaled'].values), 
            batch_size = 200_000, epochs = 20, callbacks = [ckpt, reduce_lr, es])
