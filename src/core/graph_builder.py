import numpy as np
import pandas as pd
from collections import deque, defaultdict

def haversine_m(lat1, lon1, lat2, lon2):
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    r = 6_371_000.
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def select_connected_subgraph(
    nodes,
    edges,
    seed_node,
    n_nodes,
):
    adjacency = defaultdict(list)

    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = {seed_node}
    queue = deque([seed_node])

    selected = []

    while queue and len(selected) < n_nodes:

        current = queue.popleft()
        selected.append(current)

        # Geographic ordering could be added here
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    selected_set = set(selected)

    mapping = {
        old: new
        for new, old in enumerate(selected)
    }

    new_nodes = [
        nodes[i]
        for i in selected
    ]

    new_edges = [
        (mapping[u], mapping[v])
        for u, v in edges
        if u in selected_set
        and v in selected_set
    ]

    return new_nodes, new_edges, mapping


def build_graph_from_csv(path, limit=None, radius_m=20., tolerance_factor=2.0):
    df = pd.read_csv(path).dropna(subset=["latitude", "longitude"]).copy()
    
    #If we want a random selection, here is the code. 
    #if limit is not None: 
    #    df = df.head(limit).copy()
        #seed = 1
        #df = df.sample(n=limit,random_state=seed).copy()
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
        
    df = df.reset_index(drop=True)
    xy = df[["latitude", "longitude"]].to_numpy()
    
    threshold = radius_m * tolerance_factor
    edges = []
    
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            if haversine_m(*xy[i], *xy[j]) < threshold: 
                edges.append((i, j))

 

    nodes = df[["id", "latitude", "longitude"]].to_dict("records")

    nodes, edges, mapping = select_connected_subgraph(nodes, edges, seed_node=limit, n_nodes=30)
                
    return nodes, edges