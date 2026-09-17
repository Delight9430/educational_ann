
from draw import Chart
import tensorflow as tf

class Network_wrapper:
    def __init__(self, title, num_inputs):
        self.num_inputs = num_inputs
        self.chart = Chart(title, num_inputs)
        self.epoch = 0
        self.num_input_neurons = None
        self.hidden_layers = []
        self.num_ouput_neurons = None

    def add_input_neurons(self, num_input_neurons):
        self.num_input_neurons = num_input_neurons
        self.chart.add_input_neurons(num_input_neurons)

    def add_hidden_neurons(self, num_hidden_neurons, *more_neurons):
        layers_to_add = [num_hidden_neurons] + list(more_neurons)
        for item in layers_to_add:
            if isinstance(item, (list, tuple)):
                for n in item:
                    self.hidden_layers.append(int(n))
                    self.chart.add_hidden_layer(int(n))
            else:
                self.hidden_layers.append(int(item))
                self.chart.add_hidden_layer(int(item))

    def add_output_neurons(self, num_ouput_neurons):
        self.num_ouput_neurons = num_ouput_neurons
        self.chart.add_output_neurons(num_ouput_neurons)

    def start(self):
        # TODO throw if number of inputs or outputs is not yet set
        model_layers = [
            # First hidden layer (input neurons)
            tf.keras.layers.Dense(units=self.num_input_neurons, input_dim=self.num_inputs, activation='relu')
        ]
        # Arbitrary hidden layers between input_neurons and output_neurons
        for h_units in self.hidden_layers:
            model_layers.append(tf.keras.layers.Dense(units=h_units, activation='relu'))

        # Output layer
        model_layers.append(tf.keras.layers.Dense(units=self.num_ouput_neurons, activation='sigmoid'))

        self.model = tf.keras.Sequential(model_layers)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mse')

    def iterate(self, data, key):
        if self.epoch == 0:
            self.start()
            # TODO: update if this changes each iteration
            cluster_1 = [p for p, m in zip(data, key) if m == 1.0]
            cluster_2 = [p for p, m in zip(data, key) if m == 0.0]
            self.chart.set_key(cluster_1, cluster_2)
        self.epoch += 1
        # Train for one epoch
        history = self.model.fit(data, key, epochs=1, verbose=0)
        current_loss = history.history['loss'][0]
        
        # See what the network currently predicts
        current_predictions = self.model.predict(data, verbose=0)
        # Update visualizer
        self.chart.draw_network(self.epoch, self.model.layers, data, current_predictions, current_loss)

    def end(self):
        self.chart.wait()