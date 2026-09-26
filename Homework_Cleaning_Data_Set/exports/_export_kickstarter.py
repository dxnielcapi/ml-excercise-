"""Reproduce el pipeline del notebook data-cleaning-challenge-scale-and-normalize-data.ipynb
y exporta el dataset original y el final (imputado + variables derivadas)."""
import numpy as np
import pandas as pd

RANDOM_STATE = 0
np.random.seed(0)
ORIG = "kickstarter_original.csv"

try:
    kickstarters_2017 = pd.read_csv(ORIG)
except UnicodeDecodeError:
    kickstarters_2017 = pd.read_csv(ORIG, encoding="ISO-8859-1")
print("Original:", kickstarters_2017.shape)

# --- seccion 2.1: fechas y duracion ---
ks = kickstarters_2017.copy()
ks["lanzamiento"] = pd.to_datetime(ks.launched, errors="coerce")
ks["cierre"] = pd.to_datetime(ks.deadline, errors="coerce")
ks["duracion"] = (ks.cierre - ks.lanzamiento).dt.days

fecha_imposible = ks.lanzamiento.dt.year < 2009

# --- seccion 2.3: imputacion de dominio en lugar de dropear ---
tipo_cambio = (ks.loc[ks.pledged > 0]
                 .assign(fx=lambda d: d.usd_pledged_real / d.pledged)
                 .groupby("currency").fx.median())

ks["usd_pledged_faltante"] = ks["usd pledged"].isna().astype(int)
ks["usd pledged"] = ks["usd pledged"].fillna(ks.pledged * ks.currency.map(tipo_cambio))
ks["usd pledged"] = ks["usd pledged"].fillna(ks.usd_pledged_real)

duracion_tipica = ks.loc[ks.duracion.between(1, 120)].groupby("main_category").duracion.median()
ks["fecha_imputada"] = fecha_imposible.astype(int)
reconstruido = ks.cierre - pd.to_timedelta(ks.main_category.map(duracion_tipica), unit="D")
ks.loc[fecha_imposible, "lanzamiento"] = reconstruido[fecha_imposible]
ks["duracion"] = (ks.cierre - ks.lanzamiento).dt.days

ks["nombre_faltante"] = ks.name.isna().astype(int)
ks["name"] = ks.name.fillna("(sin titulo)")
ks["nombre_largo"] = ks.name.str.len()
ks["nombre_palabras"] = ks.name.str.split().str.len()

print("Nulos restantes:", int(ks.isna().sum().sum()))
print("Filas conservadas:", f"{len(ks):,} de {len(kickstarters_2017):,}")

ks.to_csv("kickstarter_final_limpio.csv", index=False)

# --- seccion 3: subconjunto de modelado (campanas finalizadas) ---
finalizadas = ks[ks.state.isin(["successful", "failed", "canceled", "suspended"])].copy()
finalizadas["mes"] = finalizadas.lanzamiento.dt.month
finalizadas["anyo"] = finalizadas.lanzamiento.dt.year
finalizadas["log_meta"] = np.log1p(finalizadas.usd_goal_real)

VAR_NUM = ["log_meta", "duracion", "nombre_largo", "nombre_palabras", "mes", "anyo"]
VAR_CAT = ["main_category", "category", "country", "currency"]
VAR_FLAG = ["nombre_faltante"]
finalizadas[["ID", "name"] + VAR_NUM + VAR_CAT + VAR_FLAG + ["state"]].to_csv(
    "kickstarter_final_modelado.csv", index=False)

print("finalizadas:", finalizadas.shape)
