from tensorflow.keras.layers import Conv1D, Input, Bidirectional, LSTM, Concatenate, Reshape, TimeDistributed, Dense, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from tensorflow.keras.activations import softmax
import numpy as np
import tensorflow as tf 

def model_builder(index, opt, input_dim=2, output_dim=2, window_size=30, target_timestep=1):
    ''' 
    build the (index)th child model base on given param set
    '''
    input = Input(shape=(None, input_dim))
    # rnn_att = LSTM(units=input_dim, return_sequences=True, return_state=False)
    # component_att_weight = softmax(rnn_att(input), axis=-1)
    # weighted_input = tf.math.multiply(input, component_att_weight)
    # sum_input = tf.math.reduce_sum(weighted_input, axis=2, keepdims=True)

    # print(weighted_input.shape)
    # rnn_1 = Bidirectional(
    #     LSTM(units=opt['lstm']['bi_unit'][index], return_sequences=True, return_state=True, 
    #         dropout=opt['dropout'][index], recurrent_dropout=opt['dropout'][index]))

    rnn_1 = LSTM(units=opt['lstm']['bi_unit'][index] * 2, return_sequences=True, return_state=True, 
            dropout=opt['dropout'][index], recurrent_dropout=opt['dropout'][index])

    # rnn_out_1, forward_h, forward_c, backward_h, backward_c = rnn_1(input)
    rnn_out_1, state_h, state_c = rnn_1(input)
    # state_h = Concatenate(axis=-1)([forward_h, backward_h])
    # state_c = Concatenate(axis=-1)([forward_c, backward_c])

    rnn_3 = LSTM(units=opt['lstm']['si_unit'][index], return_sequences=True, return_state=False, 
                dropout=opt['dropout'][index], recurrent_dropout=opt['dropout'][index])
    rnn_out_3 = rnn_3(rnn_out_1, initial_state=[state_h, state_c])

    attention = Attention()
    context_vec = attention([rnn_out_3, state_h])
    context_and_rnn_2_out = Concatenate(axis=-1)([context_vec, rnn_out_3])
    Wc = Dense(units=128, activation=tf.math.tanh, use_bias=False)
    attention_vec = Wc(context_and_rnn_2_out)

    conv_3 = Conv1D(filters=opt['conv']['n_kernels'][index][2], activation='relu',
                kernel_size=window_size - target_timestep + 1)
    conv_out_3 = conv_3(attention_vec)

    dense_3 = TimeDistributed(Dense(units=output_dim))
    
    # print(rnn_out_3.shape)
    outs = dense_3(conv_out_3)
    # print(outs.shape)
    model = Model(inputs=input, outputs=outs)

    if opt['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=opt['lr'][index])
    elif opt['optimizer'] == 'ada':
        optimizer = Adadelta(learning_rate=opt['lr'][index])
    else:
        optimizer = Adam(learning_rate=opt['lr'][index],amsgrad=False)
    model.compile(loss=opt['loss'], optimizer=optimizer, metrics=['mae', 'mape'])

    return model


def train_model(model, index, x_train, y_train, batch_size, epochs, fraction=0.1, patience=0, early_stop=False, save_dir='', pred_factor='q'):
    callbacks = []

    checkpoint = ModelCheckpoint(save_dir + f'best_model_{pred_factor}_{index}.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    callbacks.append(checkpoint)

    #early_stop = epochs == 250
    if (early_stop):
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks.append(early_stop)

    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_split=fraction)

    return model, history

if __name__ == '__main__':
    import yaml
    with open('./settings/model/child_default.yaml', 'r') as f:
        child_config = yaml.load(f, Loader=yaml.FullLoader)
        print(child_config)
        model = model_builder(0, child_config['rnn_cnn'], input_dim=20, output_dim=1, target_timestep=7)
        print(model.summary())
        x = np.random.rand(10,30, 20)
        y = model(x)
        # fake training
        # x = np.random.rand(4000, 30, 20)
        # y = np.random.rand(4000, 7, 1)
        # model, _ = train_model(model, 0, x, y, 200, 50)
        # y_pred = model.predict(x)

        # import matplotlib.pyplot as plt 
        # plt.plot(y[:, 0, :])
        # plt.plot(y_pred[:, 0, :])
        # plt.savefig('test.png')