from sys import path
from os.path import dirname, realpath


MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)



import numpy as np
import tensorflow as tf

from tensorflow import keras



def get_and_append_data(relative_filepath):

    data = np.genfromtxt(MY_DIR + "/" + relative_filepath, delimiter=',')

    print(data.shape)
    # fill = np.ones( (data.shape[0],1) )
    # print(fill.shape)
    # data = np.concatenate( [data, fill] , axis=1)

    # data = np.insert(data, -1, 1.0, axis=1)
    # data[:, -1] -= .5
    # data[:, -1] *= 2

    return data[ : , :-2], data[ : , -1]


def get_model(input_width, num_hidden_layers, hidden_layer_width, activation, kernel_initializer, bias_initializer):



    model = keras.Sequential()

    model.add( keras.layers.Dense(hidden_layer_width, input_dim=input_width, activation=activation, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer ) )

    for i in range(1, num_hidden_layers):
        model.add(keras.layers.Dense(hidden_layer_width, activation=activation, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer))

    # Output layer: linear because it's the output
    model.add( keras.layers.Dense( 1, activation='linear', bias_initializer=bias_initializer, kernel_initializer=kernel_initializer ) )

    return model


if __name__ == '__main__':


    limited_data_filename = "/Data/limited_5000.csv"

    limited_data_x, limited_data_y = get_and_append_data(limited_data_filename)
    
    train_size = 3000

    print(limited_data_x[:10])
    print(limited_data_y[:10])

    results = "{}\t&{:<7.0f}\t&{:<7.0f}\t&{:>10.15f}\t&{:>10.15f}\t\\\\\hline"
    print("method\t&width\t&depth\t&train error\t&test error\t\\\\\hline")


    for method in [ ['tanh','glorot_normal' ], ['relu', 'he_normal'] ]:
        for width in [5, 10, 25, 50, 100]:
            for depth in [3,5,9]:
                model = get_model(input_width=limited_data_x.shape[1], num_hidden_layers=depth, hidden_layer_width=width, activation=method[0],
                                  bias_initializer=method[1], kernel_initializer=method[1])

                model.compile('adam', loss='mean_squared_error', metrics=['accuracy', 'binary_accuracy'])

                model.fit(limited_data_x[:train_size], limited_data_y[:train_size], epochs=25, batch_size=70, verbose=0)

                train_scores = model.evaluate(limited_data_x[:train_size], limited_data_y[:train_size], verbose=0)
                # print("train error: ", train_scores)
                test_scores = model.evaluate(limited_data_x[train_size:], limited_data_y[train_size:], verbose=0)

                print(results.format(method[0], width, depth, 1-train_scores[2], 1-test_scores[2]))




