import matplotlib.pyplot as plt
import networkx as nx


def create_networkx_graph(nodes, edges):
    """
    Convert the graph representation used by graph_builder.py
    into a NetworkX graph.
    """
    G = nx.Graph()

    for i, node in enumerate(nodes):
        G.add_node(
            i,
            id=node["id"],
            latitude=node["latitude"],
            longitude=node["longitude"],
        )

    G.add_edges_from(edges)

    return G


def plot_graph(
    nodes,
    edges,
    selected_bits=None,
    layout="spring",
    show_labels=True,
    figsize=(10, 8),
):
    """
    Plot the MIS graph.

    Parameters
    ----------
    nodes : list
        Nodes returned by build_graph_from_csv().

    edges : list
        Edge tuples returned by build_graph_from_csv().

    selected_bits : list[int], optional
        Binary MIS solution, for example:
        [1, 0, 1, 0, 1]

        Vertices equal to 1 are highlighted.

    layout : str
        "spring" or "geographic".

    show_labels : bool
        Show vertex indices.

    figsize : tuple
        Matplotlib figure size.
    """

    G = create_networkx_graph(nodes, edges)

    if layout == "geographic":
        # NetworkX expects (x, y):
        # x = longitude
        # y = latitude
        pos = {
            i: (
                node["longitude"],
                node["latitude"],
            )
            for i, node in enumerate(nodes)
        }

    elif layout == "spring":
        pos = nx.spring_layout(G, seed=42)

    else:
        raise ValueError(
            "layout must be 'spring' or 'geographic'"
        )

    plt.figure(figsize=figsize)

    # Draw edges first
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.5,
        width=1.5,
    )

    if selected_bits is None:
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=650,
        )

    else:
        print(selected_bits)
        input("SELECTED BITS")
        selected = [
            i for i, bit in enumerate(selected_bits)
            if bit == 1
        ]

        not_selected = [
            i for i, bit in enumerate(selected_bits)
            if bit == 0
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=not_selected,
            node_size=650,
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=selected,
            node_size=900,
            node_color = "red",
        )

    if show_labels:
        labels = {
            i: str(i)
            for i in range(len(nodes))
        }

        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=10,
        )

    plt.title(
        f"MIS graph | "
        f"|V| = {G.number_of_nodes()} | "
        f"|E| = {G.number_of_edges()}"
    )

    if layout == "geographic":
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    plt.tight_layout()
    plt.show()