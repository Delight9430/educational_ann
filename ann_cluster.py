
from network_wrapper import Network_wrapper
from data import get_two_gaussian_clusters
# TODO: move charts to a seperate 
def main():

    network = Network_wrapper("v", 2)
    network.add_input_neurons(2)
    network.add_hidden_neurons(4)
    network.add_hidden_neurons(4)
    network.add_output_neurons(1)
    
    # 4. Training Loop
    epochs = 150
    X, y = get_two_gaussian_clusters(num_points=150,
                                     noise=4)
    for epoch in range(1, epochs + 1):
        network.iterate(X, y)
    network.end()

main()