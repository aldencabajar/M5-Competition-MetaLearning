import tensorflow as tf
import numpy as np
import argparse
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
    feat_vars = dataset.columns[~dataset.columns.isin(
        ['id', 'sales', 'scaled', 'wt', 'day_of_year', 'day'])]

    # create the model    
    model = create_model(n_hidden_layers = 3, hidden_units = 500, lr = 0.00005)    

    #setup other params
    ckpt = ModelCheckpoint("weights.h5", monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=2, min_lr=0.00001)
    es = EarlyStopping(monitor='val_loss', patience=5)
    keras.backend.clear_session()
    train_data_dict = {'ts_id_input': train_id_enc, 
                    'ts_features':train_data[feat_vars]}
    val_data_dict = {'ts_id_input': validation_id_enc, 
                    'ts_features': validation_data[feat_vars]}
                    
    history = model.fit(x = train_data_dict, y = train_data['scaled'].values, 
            validation_data = (val_data_dict, validation_data['scaled'].values), 
            batch_size = 200_000, epochs = 20, callbacks = [ckpt, reduce_lr, es])
