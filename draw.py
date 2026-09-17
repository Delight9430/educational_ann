
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

if sys.platform.startswith("linux"):
    matplotlib.use('TkAgg') 

class Chart:
    def __init__(self, title, num_inputs):
        # 3. Setup the Matplotlib Visualizer
        plt.ion() # Turn on interactive mode for live updating
        fig, self.ax = plt.subplots(figsize=(10, 6))
        fig.canvas.manager.set_window_title(title)
        self.num_inputs = num_inputs
        self.num_input_neurons = None
        self.hidden_layers = []
        self.num_ouput_neurons = None

        self.input_pos_list = []
        self.input_neuron_pos_list = []
        self.hidden_neuron_pos_lists = []
        self.output_neuron_pos_list = []

        self._recalculate_positions()

    def _recalculate_positions(self):
        # Determine active layers to evenly space in the x-axis
        layer_specs = [('inputs', self.num_inputs)]
        if self.num_input_neurons is not None:
            layer_specs.append(('input_neurons', self.num_input_neurons))
        for i, count in enumerate(self.hidden_layers):
            layer_specs.append((f'hidden_{i}', count))
        if self.num_ouput_neurons is not None:
            layer_specs.append(('output_neurons', self.num_ouput_neurons))

        total_cols = len(layer_specs)
        # If output neurons are not yet added, anticipate them so current columns are placed sensibly
        if self.num_ouput_neurons is None and total_cols > 0:
            anticipated_cols = total_cols + 1
        else:
            anticipated_cols = total_cols

        start_x = 0.2
        end_x = 0.8
        if anticipated_cols > 1:
            dx = (end_x - start_x) / (anticipated_cols - 1)
        else:
            dx = 0.3

        all_counts = [count for _, count in layer_specs]
        max_neurons = max(all_counts) if all_counts else 1
        min_dy = 1.0 / (max_neurons + 1)

        # Scale element dimensions if there are many layers or neurons
        self.neuron_radius = min(0.07, dx * 0.3, min_dy * 0.3)
        self.input_side_length = self.neuron_radius * 2.0
        self.font_size = max(6, min(12, int(self.neuron_radius / 0.08 * 12)))
        self.weight_font_size = max(6, min(12, int(dx / 0.3 * 12)))

        self.hidden_neuron_pos_lists = []

        col_idx = 0
        # 1. Inputs
        input_x = start_x + col_idx * dx
        input_y_step = 1.0 / (self.num_inputs + 1)
        self.input_pos_list = [(input_x, input_y_step * i) for i in range(self.num_inputs, 0, -1)]
        col_idx += 1

        # 2. Input neurons
        if self.num_input_neurons is not None:
            input_neuron_x = start_x + col_idx * dx
            output_neuron_y_step = 1.0 / (self.num_input_neurons + 1)
            self.input_neuron_pos_list = [(input_neuron_x, output_neuron_y_step * i) for i in range(self.num_input_neurons, 0, -1)]
            col_idx += 1
        else:
            self.input_neuron_pos_list = []

        # 3. Hidden layers
        for count in self.hidden_layers:
            h_x = start_x + col_idx * dx
            h_y_step = 1.0 / (count + 1)
            h_pos_list = [(h_x, h_y_step * i) for i in range(count, 0, -1)]
            self.hidden_neuron_pos_lists.append(h_pos_list)
            col_idx += 1

        # 4. Output neurons
        if self.num_ouput_neurons is not None:
            output_neuron_x = start_x + col_idx * dx
            output_neuron_y_step = 1.0 / (self.num_ouput_neurons + 1)
            self.output_neuron_pos_list = [(output_neuron_x, output_neuron_y_step * i) for i in range(self.num_ouput_neurons, 0, -1)]
        else:
            self.output_neuron_pos_list = []

        # Aliases for compatibility
        self.hidden_pos_lists = self.hidden_neuron_pos_lists

    def add_input_neurons(self, num_input_neurons):
        self.num_input_neurons = num_input_neurons
        self._recalculate_positions()

    def add_hidden_layer(self, num_hidden_neurons, *more_neurons):
        layers_to_add = [num_hidden_neurons] + list(more_neurons)
        for item in layers_to_add:
            if isinstance(item, (list, tuple)):
                for n in item:
                    self.hidden_layers.append(int(n))
            else:
                self.hidden_layers.append(int(item))
        self._recalculate_positions()

    def add_output_neurons(self, num_ouput_neuron):
        self.num_ouput_neurons = num_ouput_neuron
        self.num_output_neurons = num_ouput_neuron
        self._recalculate_positions()

    def wait(self):
        # Keep window open when training finishes
        plt.ioff()
        plt.show()

    def setup_axis(self):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(0, 1.2)
        self.ax.set_ylim(0, 1)
        
    def draw_network(self, epoch, layers, input_points, predictions, loss):
        self._recalculate_positions()
        self.setup_axis()

        # Collect all position lists in order: Inputs -> Input Neurons -> Hidden Layers -> Output Neurons
        all_layers_pos = [self.input_pos_list]
        if self.input_neuron_pos_list:
            all_layers_pos.append(self.input_neuron_pos_list)
        for h_pos_list in self.hidden_neuron_pos_lists:
            all_layers_pos.append(h_pos_list)
        if self.output_neuron_pos_list:
            all_layers_pos.append(self.output_neuron_pos_list)

        # Draw weights between adjacent layers
        for layer_idx in range(len(all_layers_pos) - 1):
            if layer_idx < len(layers):
                weights_and_biases = layers[layer_idx].get_weights()
                if len(weights_and_biases) > 0:
                    layer_weights = weights_and_biases[0]
                    prev_pos_list = all_layers_pos[layer_idx]
                    curr_pos_list = all_layers_pos[layer_idx + 1]
                    for i, prev_pos in enumerate(prev_pos_list):
                        for j, curr_pos in enumerate(curr_pos_list):
                            self.draw_weight(layer_weights[i][j], curr_pos, prev_pos, f"w{i+1}")

        # Draw inputs
        for i, node_pos in enumerate(self.input_pos_list):
            self.draw_input(i, node_pos)

        # Draw input neurons
        for _, node_pos in enumerate(self.input_neuron_pos_list):
            self.draw_neuron(node_pos)

        # Draw hidden layer neurons
        for h_pos_list in self.hidden_neuron_pos_lists:
            for node_pos in h_pos_list:
                self.draw_neuron(node_pos)

        # Extract output biases if available
        out_biases = None
        if len(layers) > 0:
            out_weights_and_biases = layers[-1].get_weights()
            if len(out_weights_and_biases) > 1:
                out_biases = out_weights_and_biases[1]

        # Draw output neurons
        bias_offset = max(0.08, self.neuron_radius + 0.04)
        for i, neuron_pos in enumerate(self.output_neuron_pos_list):
            self.draw_neuron(neuron_pos)
            # Display Bias next to output neuron
            if out_biases is not None and i < len(out_biases):
                self.ax.text(neuron_pos[0] + bias_offset, neuron_pos[1], f"Bias: {out_biases[i]:.2f}", 
                        va='center', fontsize=self.font_size, fontweight='bold')

        self.draw_prediction(epoch, input_points, predictions, loss)

        plt.pause(0.5) 

    def draw_prediction(self, epoch, input_points, predictions, loss):
        # Title and stats
        self.ax.set_title(f"Epoch: {epoch} | Loss (MSE): {loss:.4f}", fontsize=16, loc='left')
        ax_inset = self.ax.inset_axes([0.7, 0.7, 0.4, 0.4])

        # Plot inside the inset
        for i, point in enumerate(input_points):
            my_color = (1.0 - predictions[i][0], 0.0, predictions[i][0])
            ax_inset.plot(point[0], point[1], marker="o", color=my_color)
        ax_inset.set_xlim(-8, 8)
        ax_inset.set_ylim(-8, 8)
        ax_inset.set_title("prediction", fontsize=9)

    def draw_neuron(self, pos, radius=None):
        if radius is None:
            radius = getattr(self, 'neuron_radius', 0.08)
        out_circle = plt.Circle(pos, radius, facecolor='white', edgecolor='black', linewidth=2, zorder=2)
        self.ax.add_patch(out_circle)

    def draw_input(self, i, pos, side_length=None):
        if side_length is None:
            side_length = getattr(self, 'input_side_length', 0.16)
        fontsize = getattr(self, 'font_size', 12)
        center_pos = (pos[0] - side_length/2.0, pos[1] - side_length/2.0)
        r = patches.Rectangle(center_pos, side_length, side_length, facecolor='green', edgecolor='black', linewidth=2, zorder=2)
        self.ax.add_patch(r)
        self.ax.text(pos[0], pos[1], f"Input {i+1}", ha='center', va='center', fontsize=fontsize)

    def draw_weight(self, weight, start_pos, end_pos, name, fontsize=None):
        if fontsize is None:
            fontsize = getattr(self, 'weight_font_size', 12)
        # Line thickness based on weight magnitude
        linewidth = min(abs(weight) * 2 + 1, 10) 
        # Color: Green for positive weight, Red for negative
        color = 'green' if weight > 0 else 'red'
            
        self.ax.plot([end_pos[0], start_pos[0]], [end_pos[1], start_pos[1]], 
                    color=color, linewidth=linewidth, zorder=1, alpha=0.6)
            
        # Display exact weight value in the middle of the line
        mid_x = (end_pos[0] + start_pos[0]) / 2.0
        mid_y = (end_pos[1] + start_pos[1]) / 2.0
        self.ax.text(mid_x, mid_y, f"{name}: {weight:.2f}", 
                    ha='center', va='center', fontsize=fontsize, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
