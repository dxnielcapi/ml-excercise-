"""Reproduce el pipeline del notebook wilson-editing-irisplant.ipynb
y exporta el dataset original y el final tras la edicion de Wilson (k=3)."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

iris = load_iris(as_frame=True)
df = iris.frame
class_names = iris.target_names

# --- original ---
orig = df.copy()
orig.insert(0, "fila_original", np.arange(len(df)))
orig["especie"] = [class_names[c] for c in df.target]
orig.to_csv("iris_original.csv", index=False)

# --- split 80/20 estratificado + estandarizacion (fit solo en train) ---
X = df.drop(columns="target").values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

idx_all = np.arange(len(df))
idx_train, idx_test = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=42)
assert np.allclose(X[idx_train], X_train) and np.array_equal(y[idx_train], y_train)

# --- edicion de Wilson (k=3) ---
def wilson_editing_details(A, yy, k=3):
    A, yy = np.asarray(A), np.asarray(yy)
    nn = NearestNeighbors(n_neighbors=k).fit(A)
    _, idx = nn.kneighbors()                       # sin contarse a si mismo
    neighbor_labels = yy[idx]
    votes = np.array([np.bincount(row, minlength=yy.max() + 1).argmax()
                      for row in neighbor_labels])
    return votes != yy, votes, neighbor_labels

atypical, votes, neighbor_labels = wilson_editing_details(X_train_scaled, y_train, k=3)
keep = ~atypical
feat = iris.feature_names


def guardar(nombre, A, A_scaled, yy, idx, extra=None):
    d = pd.DataFrame(A, columns=feat)
    d.insert(0, "fila_original", idx)
    d["target"] = yy
    d["especie"] = [class_names[c] for c in yy]
    for c, col in zip(feat, A_scaled.T):
        d[c.replace(" (cm)", "") + "_z"] = col
    if extra is not None:
        for k_, v in extra.items():
            d[k_] = v
    d.to_csv(nombre, index=False)
    return d


# train inicial (antes de editar), train editado (final), test intacto
guardar("iris_train_inicial.csv", X_train, X_train_scaled, y_train, idx_train,
        {"atipico_wilson": atypical.astype(int),
         "voto_3nn": [class_names[c] for c in votes],
         "etiquetas_vecinos": ["-".join(class_names[c][:4] for c in row)
                               for row in neighbor_labels]})
guardar("iris_train_editado_final.csv", X_train[keep], X_train_scaled[keep],
        y_train[keep], idx_train[keep])
guardar("iris_test.csv", X_test, X_test_scaled, y_test, idx_test)

# patrones eliminados, para el reporte
elim = guardar("iris_eliminados_wilson.csv", X_train[atypical], X_train_scaled[atypical],
               y_train[atypical], idx_train[atypical],
               {"voto_3nn": [class_names[c] for c in votes[atypical]],
                "etiquetas_vecinos": ["-".join(class_names[c][:4] for c in row)
                                      for row in neighbor_labels[atypical]]})

print(f"train inicial : {len(y_train)}")
print(f"train editado : {keep.sum()}  (eliminados {atypical.sum()}, "
      f"{atypical.sum()/len(y_train):.1%})")
print("filas eliminadas:", sorted(idx_train[atypical].tolist()))
print("por clase:", dict(zip(class_names, np.bincount(y_train[atypical], minlength=3))))
