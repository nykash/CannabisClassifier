import numpy as np
import pandas as pd

def convert_triangle_to_mat(grad):
    fc = np.zeros((90, 90))

    c = 0
    for i in range(90):
        for j in range(i + 1, 90):
            fc[i, j] = grad[c]
            c += 1

    c = 0
    fc = fc.T
    for i in range(90):
        for j in range(i + 1, 90):
            fc[i, j] = grad[c]
            c += 1

    return fc

def get_brain_connectivity_functional_connectivity(x_test, weights):
    bcs = [np.abs(convert_triangle_to_mat(x_test[i])) for i in range(x_test.shape[0])]
    fcms = [b * weights for b in bcs]

    return fcms, bcs

def threshold(x, thresh=0.02):
    x = np.copy(x)
    x[x < np.percentile(x, (1-thresh)*100)] = 0
    return x

def binarize(x):
    n = np.copy(x)
    n[n > 0] = 1

    return n

def degree_centrality(f):
    centralities = []
    for i in range(90):
        centralities.append(np.mean(f[i]))

    return centralities

def convert_centralities_to_df(centrals, indices):
    df_raw = {"person": [], "roi": [], "centrality": []}
    for i in range(centrals.shape[0]):
        for j in range(centrals.shape[1]):
            if j not in indices:
                continue
            df_raw["person"].append(i)
            df_raw["roi"].append(roi_labels.description[j])
            df_raw["centrality"].append(centrals[i, j])

    return pd.DataFrame(df_raw)

def get_sorted_order(df, indices, reverse=False):
    labs = roi_labels.description[indices]
    means = [df[df.roi == l].centrality.mean() for l in labs]
    slabs = labs.iloc[np.argsort(means)]

    slabs = reversed(slabs) if reverse else slabs

    return slabs

def cosim(x, y):
    return np.dot(x, y)/np.linalg.norm(x)/np.linalg.norm(y)

roi_labels = pd.read_table('data/stanford_coord_mapping_nonetwork.tsv')
