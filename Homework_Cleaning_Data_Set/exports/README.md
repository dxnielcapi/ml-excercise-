# Exportaciones

Generadas con `_export_kickstarter.py` y `_export_iris.py`, que reproducen
exactamente los pipelines de los notebooks.

## Tendencias de campaña (Kickstarter) — `data-cleaning-challenge-scale-and-normalize-data.ipynb`

| archivo | filas | qué es |
|---|---|---|
| `kickstarter_original.csv` | 378 661 | crudo, tal cual (ks-projects-201801.csv) |
| `kickstarter_final_limpio.csv` | 378 661 | tras fechas parseadas, imputación de dominio y variables derivadas |
| `kickstarter_final_modelado.csv` | 372 300 | subconjunto de campañas finalizadas con solo las variables del modelo |

Lo aplicado al final: `lanzamiento`/`cierre` como fechas y `duracion` en días;
`usd pledged` imputado con `pledged × tipo de cambio mediano de su moneda`
(fallback `usd_pledged_real`); fechas imposibles (< 2009) reconstruidas como
`cierre − duración mediana de su categoría`; `name` nulo → `(sin titulo)`;
indicadores `usd_pledged_faltante`, `fecha_imputada`, `nombre_faltante`;
`nombre_largo`, `nombre_palabras`. Cero filas descartadas y cero nulos.

`kickstarter_final_modelado.csv` excluye las variables con fuga
(`pledged`, `backers`, `usd_pledged_real`, `usd pledged`, `usd_pledged_faltante`)
y conserva `state` como etiqueta.

## Iris — `wilson-editing-irisplant.ipynb`

| archivo | filas | qué es |
|---|---|---|
| `iris_original.csv` | 150 | `load_iris()` completo |
| `iris_train_inicial.csv` | 120 | train 80 % estratificado, con la marca `atipico_wilson` y el voto 3-NN |
| `iris_train_editado_final.csv` | 114 | **final**: train después de la edición de Wilson (k=3) |
| `iris_test.csv` | 30 | test intacto (nunca se edita) |
| `iris_eliminados_wilson.csv` | 6 | los patrones atípicos descartados |

Split `test_size=0.2, stratify=y, random_state=42`; `StandardScaler` ajustado
solo en train (columnas `*_z` = versión estandarizada, las de cm son las
originales). Wilson k=3 elimina 6 patrones (5 %): 0 setosa, 2 versicolor,
4 virginica — filas originales 72, 83, 106, 119, 133, 149.
