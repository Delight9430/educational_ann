
from network_wrapper import Network_wrapper
from data import get_two_gaussian_clusters
from data import get_two_gaussian_circles

def main():

    network = Network_wrapper("v", 2)
    network.add_input_neurons(2)
    network.add_hidden_neurons(4)
    network.add_hidden_neurons(4)
    network.add_output_neurons(1)
    
    # 4. Training Loop
    points, group = get_two_gaussian_circles(n_points=50,
                                    noise=0.01 )
    for _ in range(1, 150):
        network.iterate(points, group)
    network.end()

main()