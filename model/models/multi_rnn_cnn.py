from tensorflow.keras.layers import Conv1D, Input, Bidirectional, LSTM, Concatenate, Reshape, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop

def model_builder(index, input_dim=2, output_dim=2, window_size=30, target_timestep=1, opt=None):
    ''' 
    build the (index)th child model base on given param set
    '''
    input = Input(shape=(None, input_dim))

    conv = Conv1D(filters=opt['conv']['n_kernels'][index][0], 
                kernel_size=opt['conv']['kernel_s'][index][0], padding='same')
    conv_out = conv(input)
    conv_2 = Conv1D(filters=opt['conv']['n_kernels'][index][1], 
                kernel_size=opt['conv']['kernel_s'][index][1], padding='same')
    conv_out_2 = conv_2(conv_out)
    conv_3 = Conv1D(filters=opt['conv']['n_kernels'][index][2], 
                kernel_size=window_size - target_timestep + 1)
    conv_out_3 = conv_3(conv_out_2)

    rnn_1 = Bidirectional(
        LSTM(units=opt['lstm']['bi_unit'][index], return_sequences=True, return_state=True, 
            dropout=opt['dropout'][index], recurrent_dropout=opt['dropout'][index]))

    rnn_out_1, forward_h, forward_c, backward_h, backward_c = rnn_1(conv_out_3)
    state_h = Concatenate(axis=-1)([forward_h, backward_h])
    state_c = Concatenate(axis=-1)([forward_c, backward_c])

    rnn_3 = LSTM(units=opt['lstm']['si_unit'][index], return_sequences=False, return_state=False, 
                dropout=opt['dropout'][index], recurrent_dropout=opt['dropout'][index])
    rnn_out_3 = rnn_3(rnn_out_1, initial_state=[state_h, state_c])

    dense_3 = Dense(units=output_dim)
    output = dense_3(rnn_out_3)

    model = Model(inputs=input, outputs=output)

    if opt['optimizer'] == 'rmsprop':
        optimizer = RMSprop(learning_rate=opt['lr'][index])
    elif opt['optimizer'] == 'ada':
        optimizer = Adadelta(learning_rate=opt['lr'][index])
    else:
        optimizer = Adam(learning_rate=opt['lr'][index],amsgrad=False)
    model.compile(loss=opt['loss'], optimizer=optimizer, metrics=['mae', 'mape'])

    return model


def train_model(model, index, x_train, y_train, batch_size, epochs, fraction=0.1, patience=0, early_stop=False, save_dir=''):
    callbacks = []

    checkpoint = ModelCheckpoint(save_dir + f'best_model_{index}.hdf5',
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
