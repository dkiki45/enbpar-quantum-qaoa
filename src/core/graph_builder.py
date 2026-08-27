import numpy as np
import pandas as pd

def haversine_m(lat1, lon1, lat2, lon2):
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    r = 6_371_000.
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin(dp/2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl/2)**2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def build_graph_from_csv(path, limit=None, radius_m=20., tolerance_factor=2.0):
    df = pd.read_csv(path).dropna(subset=["latitude", "longitude"]).copy()
    if limit is not None: 
        df = df.head(limit).copy()
        
    df = df.reset_index(drop=True)
    xy = df[["latitude", "longitude"]].to_numpy()
    
    threshold = radius_m * tolerance_factor
    edges = []
    
    for i in range(len(xy)):
        for j in range(i+1, len(xy)):
            if haversine_m(*xy[i], *xy[j]) < threshold: 
                edges.append((i, j))
                
    return df.to_dict("records"), edges