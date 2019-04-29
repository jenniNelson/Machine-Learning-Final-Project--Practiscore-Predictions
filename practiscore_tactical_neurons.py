from sys import path
from os.path import dirname, realpath


MY_DIR = dirname(realpath(__file__))
path.append(MY_DIR)
PARENT_DIR = dirname(path[0])
path.append(PARENT_DIR)



import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import initializers as inits

class ControlPanel:
    def __init__(self, division='limited', data_x=None, data_y=None, model=None, history_size = 5, activation='tanh', initialization=None,
                 network_type='NN', width=5, depth=10, epochs = 25, batch_size=70,
                 train_test_ratio=.5, label_index=-1, data_transform_func = None,
                 powerfactor_included =True, match_size_included = True, verbose=0, print_results=True):
        self.division = division
        self.history_size = history_size
        self.activation = activation
        self.initialization = initialization
        self.network_type = network_type
        self.label_index = label_index
        self.data_transform_func = data_transform_func
        self.powerfactor_included = powerfactor_included
        self.match_size_included = match_size_included
        self.width = width
        self.depth = depth
        self.train_test_ratio = train_test_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.model= model
        self.verbose = verbose
        self.print_results = print_results
        self.data_x = data_x
        self.data_y = data_y

    def train_and_test(self):

        if self.data_x is None or self.data_y is None:
            data_filename = "Data/" + self.division + "_" + str(self.history_size) + "_5000.csv"

            self.data_x, self.data_y = read_data(data_filename, self.label_index)

        if self.data_transform_func is not None:
            self.data_x, self.data_y = self.data_transform_func(self.data_x, self.data_y)

        train_size =  int(self.data_x.shape[0] * self.train_test_ratio)

        if self.verbose == 1:
            print(self.data_x[:10])
            print(self.data_y[:10])

        train_data_x = self.data_x[:train_size]
        train_data_y = self.data_y[:train_size]

        test_data_x = self.data_x[train_size:]
        test_data_y = self.data_y[train_size:]

        if self.model is None:

            self.model = get_model(input_width=data_x.shape[1],
                                   num_hidden_layers=self.depth,
                                   hidden_layer_width=self.width,
                                   activation=self.activation,
                                   bias_initializer=self.initialization,
                                   kernel_initializer=self.initialization)

            metrics = ['accuracy', 'binary_accuracy'] if self.label_index == -1 else ['accuracy']
            self.model.compile('adam', loss='mean_squared_error', metrics=['accuracy', 'binary_accuracy'])


        if self.verbose == 1:
            self.model.summary(print_fn=print)
            print(self.model.get_config())
        self.model.fit(train_data_x, train_data_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        train_scores = self.model.evaluate(train_data_x, train_data_y, verbose=self.verbose)
        # print("train error: ", train_scores)
        test_scores = self.model.evaluate(test_data_x, test_data_y, verbose=self.verbose)


        if self.print_results:
            format_string = "initializer: {}\tactivation: {}\twidth: {}\tdepth: {}\ttrain: loss {:<7.4f}, accuracy {:<7.4f}\ttest: loss {:<7.4f}, accuracy {:<7.4f}\t"
            print(format_string.format(self.initialization, self.activation, self.width, self.depth, train_scores[0], train_scores[-1], test_scores[0], test_scores[-1]))
        return train_scores, test_scores, self.model

        # print(results.format(self.activation, self.width, self.depth, 1-train_scores[2], 1-test_scores[2]))




def read_data(relative_filepath, label_index=-1):

    data = np.genfromtxt(MY_DIR + "/" + relative_filepath, delimiter=',')

    # print(data.shape)
    # fill = np.ones( (data.shape[0],1) )
    # print(fill.shape)
    # data = np.concatenate( [data, fill] , axis=1)

    # data = np.insert(data, -1, 1.0, axis=1)
    # data[:, -1] -= .5
    # data[:, -1] *= 2

    return data[ : , :label_index], data[ : , label_index]


def get_model(input_width, num_hidden_layers, hidden_layer_width, activation, kernel_initializer=None, bias_initializer=None):



    model = keras.Sequential()

    model.add( keras.layers.Dense(hidden_layer_width, input_dim=input_width, activation=activation, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer ) )

    for i in range(1, num_hidden_layers):
        model.add(keras.layers.Dense(hidden_layer_width, activation=activation, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer))

    # Output layer: linear because it's the output
    model.add( keras.layers.Dense( 1, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer ) )

    return model



if __name__ == '__main__':


    max_test_acc = 0
    min_test_loss = 1
    max_acc_init, max_acc_act, max_acc_width, max_acc_depth = None, None, None, None
    min_loss_init, min_loss_act, min_loss_width, min_loss_depth = None, None, None, None

    data_x, data_y = read_data("Data/limited_5_5000.csv")

    for activation in [ 'tanh', 'relu', 'softmax', 'selu', 'softsign', 'sigmoid', 'linear' ]:
        for initializer in ['glorot_normal', 'he_normal', 'random_uniform']:
            for width in [5, 10, 25, 50, 100]:
                for depth in [3,5,9,11]:
                    control_panel = ControlPanel(data_x=data_x, data_y=data_y, history_size=5, activation=activation, width=width, depth=depth, verbose=0, initialization=initializer)
                    _, test_scores, _ = control_panel.train_and_test()
                    if test_scores[-1] > max_test_acc:
                        max_test_acc = test_scores[-1]
                        max_acc_init, max_acc_act, max_acc_width, max_acc_depth = initializer, activation, width, depth

                    if test_scores[0] < min_test_loss:
                        max_test_loss = test_scores[0]
                        min_loss_init, min_loss_act, min_loss_width, min_loss_depth = initializer, activation, width, depth

    print("Max accuracy settings: ", max_acc_init, max_acc_act, max_acc_width, max_acc_depth)
    print("Min loss settings: ", min_loss_init, min_loss_act, min_loss_width, min_loss_depth)





