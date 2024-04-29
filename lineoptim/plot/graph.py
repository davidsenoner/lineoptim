import networkx as nx
from networkx import Graph
import matplotlib.pyplot as plt


def get_xy_position(position, level=0):
    """
    Get x, y position based to line level. Rotate 45 degrees for each level
    """
    import numpy as np
    alpha = level * np.pi / 4
    x = int(position * np.cos(alpha))
    y = int(position * np.sin(alpha))
    return x, y


def add_line_to_graph(graph, line, offset=(0, 0), level=0):
    """
    Add line to graph
    """
    loads = line['loads']

    conductor_properties = _get_conductor_properties(line, level)

    if level > 0:
        graph.add_edge(line['name'], loads[0]['name'], **conductor_properties)

    for load, load_1 in zip(loads, loads[1:]):
        x, y = get_xy_position(load['position'], level)
        pos_load = (x + offset[0], y + offset[1])
        x, y = get_xy_position(load_1['position'], level)
        pos_load_1 = (x + offset[0], y + offset[1])

        load_properties = _get_load_properties(load)
        load_1_properties = _get_load_properties(load_1)

        graph.add_node(load['name'], pos=pos_load, **load_properties)
        graph.add_node(load_1['name'], pos=pos_load_1, **load_1_properties)
        graph.add_edge(load['name'], load_1['name'], **conductor_properties)

        if load_1.get('loads'):
            add_line_to_graph(graph, load_1, offset=pos_load_1, level=level + 1)


def _get_conductor_properties(line, level):
    impedance = line['resistivity'] + line['reactance'] * 1j
    mean_impedance = round(abs(impedance).mean().item(), 3)

    conductor_properties = {
        'name': line['name'],
        'impedance': impedance,
        'mean_impedance': mean_impedance,
        'resistivity': line['resistivity'],
        'reactance': line['reactance'],
        'level': level,
        'weight': 1 / mean_impedance
    }
    return conductor_properties


def _get_load_properties(load):
    return {
        'name': load['name'],
        'active_power': load['active_power'],
        'apparent_power': load['apparent_power'],
        'v_nominal': load['v_nominal'],
        'power_factor': load['power_factor'],
        'position': load['position'],
        'cores': load['cores'],
    }


class PlotGraph(Graph):
    def __init__(self, line, **attr):
        super().__init__(**attr)
        self.line = line
        self._weight_category_qnty = 6

        self.figure = plt.figure(figsize=(15, 10))
        self.ax = self.figure.add_subplot(111)

    @property
    def weight_category_qnty(self):
        """ Quantity of weight categories meaning the number of different line thicknesses """
        return self._weight_category_qnty

    @weight_category_qnty.setter
    def weight_category_qnty(self, value):
        self._weight_category_qnty = value

    def save(self, filename='power_line_graph.pdf'):
        # save figure to pdf
        self.figure.savefig(filename, format='pdf')

    def plot(self, line=None):

        self.clear()  # clear graph

        # use line from argument or self.line
        if line is not None:
            add_line_to_graph(self, line)
        else:
            add_line_to_graph(self, self.line)

        pos = nx.get_node_attributes(self, 'pos')

        nx.draw_networkx_nodes(self, pos, node_size=1500, node_shape='o', ax=self.ax)

        node_labels = {
            "labels": nx.get_node_attributes(self, 'name'),
            "font_size": 8,
            "font_family": "sans-serif",
            "font_color": "black",
        }

        nx.draw_networkx_labels(self, pos, ax=self.ax, **node_labels)

        edge_widths = {i: [] for i in range(self._weight_category_qnty)}
        for u, v, d in self.edges(data=True):
            category = min(int(d["weight"] // 2.5), self._weight_category_qnty)  # weight category id
            edge_widths[category].append((u, v))

        for category, edges in edge_widths.items():
            nx.draw_networkx_edges(self, pos, edgelist=edges, width=2 * (category + 1), edge_color='grey', style='-', ax=self.ax)

        edge_labels = {
            "edge_labels": nx.get_edge_attributes(self, "mean_impedance"),
            "font_size": 8,
            "font_family": "sans-serif",
            "font_color": "black",
        }
        nx.draw_networkx_edge_labels(self, pos, ax=self.ax, **edge_labels)

        # title
        self.ax.set_title("Power line graph", fontsize=14)  # title

        self.figure.show()
