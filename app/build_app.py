"""Genera la aplicación de la Entrega 4: docs/index.html (+ docs/data/partidos.json).

Toma lo que produce el pipeline (Silver y auditoría del DAG) y lo que dejó el notebook de modelado
(include/data/models/underdog_modelo.json y resultados_e3.json), y arma una página estática con gráficos
de Altair. GitHub Pages la publica desde la carpeta docs/ del repositorio.

    .venv/Scripts/python app/build_app.py

Si los datos cambian: correr el DAG, re-ejecutar el notebook y correr este script de nuevo.
"""

from __future__ import annotations

import html
import json
import sys
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from include.config import AUDIT_DIR, DATA_DIR, SILVER_DATASET_PATH, SILVER_MANIFEST_PATH  # noqa: E402
from include.src.columns import FEATURE_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN  # noqa: E402
from include.src.features import logit  # noqa: E402

SALIDA = RAIZ / "docs"
MODELOS_DIR = DATA_DIR / "models"
TEST_TEMPORADA = "2015/2016"
APUESTA = 100

# Versiones exactas de la librería que dibuja los gráficos de Altair en el navegador
VEGA, VEGA_LITE, VEGA_EMBED = "6.2.0", "6.4.1", "7.0.2"

# Los mismos colores que el notebook, con el mismo significado en toda la aplicación.
# Paleta pensada para daltonismo (familia Okabe-Ito): el verde azulado del modelo y el bermellón de "apostar a todos"
# se distinguen también sin ver rojo y verde, y las líneas llevan además trazo distinto y rótulo directo.
COLOR = {"modelo": "#0f8b7d", "modelo_suave": "#7cc2b6", "mercado": "#e08a1e", "todos": "#d55e00", "gano": "#2563a8",
         "no_gano": "#2f3542", "gano_suave": "#a9c1e0", "fondo_puntos": "#d5dae1", "tinta": "#1e2430", "tinta_suave": "#6b7380",
         "once": "#cc79a7", "azar": "#d5dae1", "empate": "#b8bec7"}
RESULTADO = alt.Scale(domain=["Ganó el underdog", "No ganó"], range=[COLOR["gano"], COLOR["no_gano"]])
UMBRALES_SIN_FORMULA = [0.15, 0.20, 0.25, 0.30, 0.35]      # candidatos del "modelo sin calibrar", elegidos en validación
CORTES_RANGO = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]  # bordes del dummy "rango de cuota", elegido en validación
MINIMO_APUESTAS_RANGO = 1000                                # un rango con menos apuestas en validación no se considera
SORTEOS_AZAR = 500
MINIMO_APUESTAS_CURVA = 20                                  # cada curva del retorno acumulado arranca con 20 apuestas

NOMBRES_GAPS = {
    "nofav_xi_overall_mean_gap": "Rating medio del XI", "nofav_top3_overall_gap": "Rating de los 3 mejores",
    "nofav_best_overall_gap": "Mejor jugador", "nofav_worst_overall_gap": "Peor titular",
    "nofav_def_overall_gap": "Rating de la defensa", "nofav_mid_overall_gap": "Rating del mediocampo",
    "nofav_att_overall_gap": "Rating del ataque", "nofav_best_finishing_gap": "Mejor definición",
    "nofav_best_reactions_gap": "Mejores reacciones", "nofav_best_marking_gap": "Mejor marca",
}
LIGAS = {
    "England Premier League": "Inglaterra", "France Ligue 1": "Francia", "Germany 1. Bundesliga": "Alemania",
    "Italy Serie A": "Italia", "Spain LIGA BBVA": "España", "Netherlands Eredivisie": "Países Bajos",
    "Portugal Liga ZON Sagres": "Portugal", "Scotland Premier League": "Escocia", "Belgium Jupiler League": "Bélgica",
}


# ----------------------------------------------------------------------------- datos y modelo
def cargar() -> tuple[pd.DataFrame, dict, dict]:
    ficha = json.loads((MODELOS_DIR / "underdog_modelo.json").read_text(encoding="utf-8"))
    resultados = json.loads((MODELOS_DIR / "resultados_e3.json").read_text(encoding="utf-8"))
    df = (pd.read_parquet(SILVER_DATASET_PATH)
            .merge(pd.read_parquet(AUDIT_DIR / "match_audit.parquet"), on="match_id")
            .sort_values(["fecha", "match_id"]).reset_index(drop=True))
    df["cuota_underdog"] = np.where(df["equipo_favorito"] == "local", df["odds_away"], df["odds_home"])
    df["prob_modelo"] = predecir(df, ficha)
    return df, ficha, resultados


def predecir(df: pd.DataFrame, ficha: dict) -> np.ndarray:
    """La fórmula del modelo, tal cual la escribió el notebook (y la misma que corre la página en JavaScript)."""
    z = np.full(len(df), ficha["intercepto"])
    for col, v in ficha["probabilidades"].items():
        z += v["coef"] * logit(df[col].fillna(v["mediana"]))
    for col, v in ficha["numericas"].items():
        z += v["coef"] * (df[col].fillna(v["mediana"]) - v["media"]) / v["desvio"]
    return 1 / (1 + np.exp(-z))


def verificar_contra_el_pipeline(df: pd.DataFrame) -> float:
    """Lo que muestra la app tiene que coincidir con el modelo del notebook."""
    import joblib
    modelo = joblib.load(MODELOS_DIR / "underdog_modelo.joblib")
    return float(np.max(np.abs(modelo.predict_proba(df)[:, 1] - df["prob_modelo"].values)))


def fuera_de_muestra(df: pd.DataFrame, resultados: dict) -> pd.DataFrame:
    """Los partidos que se predijeron sin haberlos visto, con esa predicción en `p`:
    la validación temporal (cada uno lo predice un modelo entrenado sólo con los anteriores, exportado por el notebook)
    y el test (el modelo final, que nunca vio 2015/16). Los de 2008–2010 sólo sirvieron para entrenar y quedan afuera."""
    oof = pd.Series(list(resultados["oof_modelo"].values()), index=resultados["oof_indices_match_id"])
    val = df[df["match_id"].isin(oof.index)].assign(p=lambda d: d["match_id"].map(oof), conjunto="validación temporal (2010–2015)")
    test = df[df["temporada"] == TEST_TEMPORADA].assign(p=lambda d: d["prob_modelo"], conjunto="test 2015/16 (modelo final)")
    return pd.concat([val, test]).sort_values(["fecha", "match_id"]).reset_index(drop=True)


def partidos_para_explorar(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por partido fuera de muestra, con lo que habría dicho y hecho el modelo con lo que sabía hasta ese momento."""
    favorito_local = df["equipo_favorito"] == "local"
    valor = df["p"] * df["cuota_underdog"] - 1
    apuesta = valor > 0
    gano = df[TARGET_COLUMN].astype(int)
    return pd.DataFrame({
        "fecha": df["fecha"].dt.strftime("%Y-%m-%d"),
        "temporada": df["temporada"],
        "liga": df["liga"].map(LIGAS).fillna(df["liga"]),
        "conjunto": df["conjunto"],
        "favorito": np.where(favorito_local, df["equipo_local"], df["equipo_visitante"]),
        "underdog": np.where(favorito_local, df["equipo_visitante"], df["equipo_local"]),
        "partido": df["equipo_local"] + " " + df["goles_local"].astype(str) + "–" + df["goles_visitante"].astype(str) + " " + df["equipo_visitante"],
        "resultado": np.where(gano == 1, "Ganó el underdog", "No ganó"),
        "gano": gano,
        "prob_mercado": df["prob_no_favorito"].round(4),
        "prob_modelo": df["p"].round(4),
        "cuota": df["cuota_underdog"].round(2),
        "valor": valor.round(4),
        "decision": np.where(apuesta, "Apuesta", "No apuesta"),
        "apuesta": apuesta.astype(int),
        "ganancia": np.where(apuesta, APUESTA * (df["cuota_underdog"] * gano - 1), 0).round(1),
        "gap_xi": df["nofav_xi_overall_mean_gap"].round(2),
    })


# ----------------------------------------------------------------------------- gráficos del tallo
def miles(x: float) -> str:
    return f"{x:+,.1f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


_capas_escritorio = iter(range(1, 10**6))


def escritorio() -> str:
    """Nombre único para una capa de rótulos decorativos: la página la saca en pantallas angostas (ver plantilla)."""
    return f"escritorio_{next(_capas_escritorio)}"


def es_test(e: pd.DataFrame) -> pd.Series:
    return e["conjunto"].str.startswith("test")


def coma(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def signo_pct(x: float, d: int = 1) -> str:
    """Un retorno en %, siempre con signo y coma decimal: +6,6% / −7,7%."""
    return f"{x:+.{d}f}%".replace(".", ",").replace("-", "−")


# ----------------------------------------------------------------------------- 0 · el problema
ROLES = ["Favorito", "Empate", "Underdog", "Margen de la casa"]


def grafico_problema(c: dict) -> tuple[alt.Chart, dict]:
    """Un partido real explicado de punta a punta: de las cuotas a las probabilidades, y el margen que se queda la casa."""
    cuotas = dict(zip(["local", "empate", "visitante"], c["cuotas"]))
    inv = {k: 1 / v for k, v in cuotas.items()}
    suma = sum(inv.values())
    local_favorito = inv["local"] >= inv["visitante"]
    rol = {"local": "Favorito" if local_favorito else "Underdog", "empate": "Empate",
           "visitante": "Underdog" if local_favorito else "Favorito"}
    equipo = {"local": c["local"], "empate": "empate", "visitante": c["visitante"]}
    filas_y = ["Según las cuotas (1 ÷ cuota)", "Sin el margen de la casa"]
    filas = []
    for fila, normalizar in zip(filas_y, [False, True]):
        ini = 0.0
        for k in ("local", "empate", "visitante"):
            p = inv[k] / suma if normalizar else inv[k]
            filas.append({"fila": fila, "rol": rol[k], "ini": ini, "fin": ini + p, "medio": ini + p / 2,
                          "t": f"{coma(100 * p)}%", "detalle": f"{equipo[k]}: cuota {coma(cuotas[k], 2)} → {coma(100 * p)}%"})
            ini += p
    filas.append({"fila": filas_y[0], "rol": "Margen de la casa", "ini": 1.0, "fin": suma, "medio": suma,
                  "t": f"suman {coma(100 * suma)}%",
                  "detalle": f"Las tres probabilidades suman {coma(100 * suma)}%"})
    t = pd.DataFrame(filas)
    escala = alt.Scale(domain=ROLES, range=[COLOR["no_gano"], COLOR["empate"], COLOR["gano"], COLOR["todos"]])
    y = alt.Y("fila:N", sort=filas_y, title=None, axis=alt.Axis(labelFontSize=12, labelFontWeight="bold", labelLimit=170,
                                                               labelExpr="split(datum.label, ' (')[0]"))
    x = alt.X("ini:Q", title=None, scale=alt.Scale(domain=[0, 1.04]), axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1]))
    barras = alt.Chart(t).mark_bar(height=46).encode(
        y=y, x=x, x2="fin:Q", color=alt.Color("rol:N", scale=escala, title=None,
                                              legend=alt.Legend(orient="top", columns=4, labelFontSize=12, symbolSize=180, labelLimit=140)),
        tooltip=[alt.Tooltip("rol:N", title=""), alt.Tooltip("detalle:N", title="")])
    adentro = alt.Chart(t[t["rol"] != "Margen de la casa"]).transform_filter("datum.fin - datum.ini > 0.3").mark_text(
        color="white", fontWeight="bold", fontSize=14).encode(y=y, x="medio:Q", text="t:N")
    adentro_chico = alt.Chart(t[t["rol"] != "Margen de la casa"]).transform_filter("datum.fin - datum.ini > 0.08 && datum.fin - datum.ini <= 0.3").mark_text(
        color="white", fontWeight="bold", fontSize=13).encode(y=y, x="medio:Q", text="t:N").properties(name=escritorio())
    margen = alt.Chart(t[t["rol"] == "Margen de la casa"]).mark_text(align="right", dy=-32, dx=4, fontSize=12, fontWeight="bold",
                                                                     color=COLOR["todos"]).encode(y=y, x="medio:Q", text="t:N")
    cien = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color=COLOR["tinta"], strokeDash=[4, 3]).encode(x="x:Q")
    datos = {"local": c["local"], "visitante": c["visitante"], "cuotas": c["cuotas"], "suma": suma,
             "favorito": c["local"] if local_favorito else c["visitante"], "underdog": c["visitante"] if local_favorito else c["local"],
             "p_underdog": (inv["visitante"] if local_favorito else inv["local"]) / suma,
             "cuota_underdog": cuotas["visitante"] if local_favorito else cuotas["local"],
             "p_favorito": (inv["local"] if local_favorito else inv["visitante"]) / suma}
    return (barras + adentro + adentro_chico + margen + cien).properties(width="container", height=170), datos


# ----------------------------------------------------------------------------- 1 · los datos
ETIQUETAS_DATOS = ["hasta 40%", "40–50%", "50–60%", "60–70%", "70–80%", "más de 80%"]
ETIQUETAS_DATOS_CORTAS = "{'hasta 40%': '≤40%', 'más de 80%': '>80%'}[datum.label] || replace(datum.label, '%', '')"


def tabla_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Cuánto gana el underdog según qué tan favorito es el favorito (una fila por barra)."""
    cortes = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
    t = (df.assign(tramo=pd.cut(df["prob_favorito"], cortes, labels=ETIQUETAS_DATOS))
           .groupby("tramo", observed=True)[TARGET_COLUMN].agg(tasa="mean", partidos="size").reset_index())
    t["tramo"] = t["tramo"].astype(str)
    return t


def grafico_datos(t: pd.DataFrame, tasa_media: float) -> tuple[alt.Chart, dict]:
    t = t[["tramo", "tasa", "partidos"]].copy()
    t["frase"] = (100 * t["tasa"]).round().astype(int).astype(str) + " de cada 100"
    t["n"] = t["partidos"].map(lambda n: f"{n:,}".replace(",", ".") + " partidos")
    t["destacado"] = [i in (0, len(t) - 1) for i in range(len(t))]
    # el número va más arriba en las barras cortas (arriba de su frase) y en las que quedan justo debajo de la línea del promedio
    t["subir"] = (t["tasa"] < 0.06) | ((t["tasa"] < tasa_media) & (t["tasa"] > tasa_media - 0.03))
    x = alt.X("tramo:O", sort=ETIQUETAS_DATOS, title="Chance del favorito según la cuota  (← parejo · abrumador →)",
              axis=alt.Axis(labelAngle=0, labelFontSize=12, labelFontWeight="bold", titlePadding=10, labelExpr=ETIQUETAS_DATOS_CORTAS))
    y = alt.Y("tasa:Q", title="Cuántos de cada 100 ganó el underdog", axis=alt.Axis(format="%", tickCount=4),
              scale=alt.Scale(domain=[0, 0.42]))
    base = alt.Chart(t).encode(x=x, y=y, tooltip=[
        alt.Tooltip("tramo:O", title="Chance del favorito"), alt.Tooltip("tasa:Q", title="Ganó el underdog", format=".1%"),
        alt.Tooltip("partidos:Q", title="Partidos", format=",")])
    barras = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, width={"band": 0.72}).encode(
        color=alt.condition("datum.destacado", alt.value(COLOR["gano"]), alt.value(COLOR["gano_suave"])))
    numero = base.transform_filter("!datum.subir").mark_text(dy=-10, fontSize=16, fontWeight="bold", color=COLOR["tinta"]).encode(
        text=alt.Text("tasa:Q", format=".0%"))
    numero_corto = base.transform_filter("datum.subir").mark_text(dy=-26, fontSize=16, fontWeight="bold", color=COLOR["tinta"]).encode(
        text=alt.Text("tasa:Q", format=".0%"))
    frase = base.transform_filter("datum.tasa >= 0.06").mark_text(baseline="top", dy=8, fontSize=11, color="white").encode(
        text="frase:N").properties(name=escritorio())
    frase_corta = base.transform_filter("datum.tasa < 0.06").mark_text(dy=-8, fontSize=11, color=COLOR["tinta_suave"]).encode(
        text="frase:N").properties(name=escritorio())
    cantidad = alt.Chart(t).mark_text(baseline="bottom", dy=-6, fontSize=10.5, fontWeight="bold", color="white").encode(
        x=x, y=alt.value(320), text="n:N").properties(name=escritorio())
    promedio = pd.DataFrame({"tasa": [tasa_media], "t": [f"promedio de todos los partidos: {round(100 * tasa_media)} de cada 100"]})
    regla = alt.Chart(promedio).mark_rule(color=COLOR["tinta"], strokeDash=[5, 4], strokeWidth=1.5).encode(y="tasa:Q")
    rotulo = alt.Chart(promedio).mark_text(align="right", x="width", dx=-2, dy=-8, fontSize=12, fontWeight="bold",
                                           color=COLOR["tinta"]).encode(y="tasa:Q", text="t:N").properties(name=escritorio())
    return ((barras + regla + numero + numero_corto + frase + frase_corta + cantidad + rotulo).properties(width="container", height=320),
            {"alto": t["tasa"].iloc[0], "bajo": t["tasa"].iloc[-1]})


# ----------------------------------------------------------------------------- 2 · lo que encontramos
HALLAZGO_REAL, HALLAZGO_CUOTA = "Lo que pasó", "Lo que esperaba la cuota"


def tabla_hallazgo(df: pd.DataFrame) -> pd.DataFrame:
    q = pd.qcut(df["nofav_xi_overall_mean_gap"], 5)
    etiquetas = [f"{i.left:+.1f} a {i.right:+.1f}".replace(".", ",") for i in q.cat.categories]
    etiquetas[0] = f"{df['nofav_xi_overall_mean_gap'].min():+.1f} a {q.cat.categories[0].right:+.1f}".replace(".", ",")
    t = (df.assign(quintil=q.cat.rename_categories(etiquetas))
           .groupby("quintil", observed=True)
           .agg(**{HALLAZGO_REAL: (TARGET_COLUMN, "mean"), HALLAZGO_CUOTA: ("prob_no_favorito", "mean"), "partidos": ("match_id", "size")})
           .reset_index().melt(id_vars=["quintil", "partidos"], var_name="serie", value_name="valor"))
    t["quintil"] = t["quintil"].astype(str)
    return t


def leyenda_lineas(orient: str = "top-left") -> alt.Legend:
    """Leyenda de líneas: muestra el trazo (continuo o punteado) igual que en el gráfico."""
    return alt.Legend(title=None, orient=orient, direction="vertical", labelFontSize=13, labelFontWeight="bold", labelLimit=320,
                      symbolType="stroke", symbolStrokeWidth=3.5, symbolSize=600, fillColor="rgba(255,255,255,.92)", padding=8)


def grafico_hallazgo(t: pd.DataFrame) -> tuple[alt.Chart, dict]:
    etiquetas = list(dict.fromkeys(t["quintil"]))
    series = [HALLAZGO_REAL, HALLAZGO_CUOTA]
    # color y trazo codifican la misma serie: así la leyenda muestra la línea punteada igual que el gráfico
    escala = alt.Scale(domain=series, range=[COLOR["gano"], COLOR["mercado"]])
    trazo = alt.Scale(domain=series, range=[[1, 0], [7, 5]])
    x = alt.X("quintil:O", sort=etiquetas, title="Rating del XI: underdog − favorito (5 grupos iguales)",
              axis=alt.Axis(labelAngle=0, labelFontSize=11, labelExpr="[split(datum.label, ' a ')[0] + ' a', split(datum.label, ' a ')[1]]"))
    y = alt.Y("valor:Q", title="Cuántos de cada 100 ganó el underdog", axis=alt.Axis(format="%", tickCount=4),
              scale=alt.Scale(domain=[0, 0.34]))
    lineas = alt.Chart(t).mark_line(point=alt.OverlayMarkDef(size=90, filled=True), strokeWidth=3.5).encode(
        x=x, y=y, color=alt.Color("serie:N", scale=escala, legend=leyenda_lineas()),
        strokeDash=alt.StrokeDash("serie:N", scale=trazo, legend=leyenda_lineas()),
        tooltip=[alt.Tooltip("quintil:O", title="Diferencia de rating"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("valor:Q", title="Valor", format=".1%"), alt.Tooltip("partidos:Q", title="Partidos", format=",")])
    lo_real = t[t["serie"] == HALLAZGO_REAL]["valor"].reset_index(drop=True)
    nota = alt.Chart(pd.DataFrame({"quintil": [etiquetas[3]], "valor": [0.07],
                                   "t": ["las dos líneas casi se pisan: el precio ya traía la calidad del once"]})).mark_text(
        fontSize=13, fontStyle="italic", color=COLOR["tinta_suave"]).encode(x=x, y=y, text="t:N").properties(name=escritorio())
    return (lineas + nota).properties(width="container", height=320), {"bajo": lo_real.iloc[0], "alto": lo_real.iloc[-1]}


# ----------------------------------------------------------------------------- 3 · ¿el modelo predice? (sin plata)
def metricas_prediccion(e: pd.DataFrame) -> dict:
    """Qué tan bien predice el modelo en el test, antes de hablar de plata, contra la cuota sola y contra un piso."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    val, test = e[~es_test(e)], e[es_test(e)]
    base = float(val["gano"].mean())       # la tasa de victorias del underdog que se conocía antes del test
    piso = np.full(len(test), base)
    filas = []
    for nombre, p, auc in [("Modelo (cuota + 10 ratings)", test["prob_modelo"], None),
                           ("La cuota sola", test["prob_mercado"], None),
                           (f"Piso: decir siempre {round(100 * base)}%", piso, 0.5)]:
        filas.append({"": nombre, "AUC: ¿ordena? (más es mejor)": auc if auc is not None else roc_auc_score(test["gano"], p),
                      "log-loss (menos es mejor)": log_loss(test["gano"], p), "Brier (menos es mejor)": brier_score_loss(test["gano"], p)})
    umbral = round(base, 2)
    dice = test["prob_modelo"] >= umbral
    gano = test["gano"] == 1
    matriz = {"vp": int((gano & dice).sum()), "fn": int((gano & ~dice).sum()), "fp": int((~gano & dice).sum()), "vn": int((~gano & ~dice).sum())}
    return {"filas": filas, "base": base, "umbral": umbral, "matriz": matriz,
            "tasa_test": float(test["gano"].mean()), "n_test": len(test), "n_val": len(val), "tasa_val": base}


def grafico_calibracion(e: pd.DataFrame) -> alt.Chart:
    """¿Cuando el modelo dice 30%, gana el underdog 3 de cada 10? Partidos del test en 10 grupos según lo que se dijo."""
    test = e[es_test(e)]
    series = ["Modelo", "La cuota"]
    partes = []
    for serie, col in zip(series, ["prob_modelo", "prob_mercado"]):
        g = (test.assign(grupo=pd.qcut(test[col], 10, labels=False, duplicates="drop"))
                 .groupby("grupo").agg(dijo=(col, "mean"), paso=("gano", "mean"), partidos=("gano", "size")).reset_index())
        partes.append(g.assign(serie=serie))
    t = pd.concat(partes, ignore_index=True)
    escala = alt.Scale(domain=series, range=[COLOR["modelo"], COLOR["mercado"]])
    trazo = alt.Scale(domain=series, range=[[1, 0], [7, 5]])
    x = alt.X("dijo:Q", title="Lo que se dijo: chance del underdog (promedio del grupo)",
              axis=alt.Axis(format="%", tickCount=5), scale=alt.Scale(domain=[0, 0.45]))
    y = alt.Y("paso:Q", title="Lo que pasó: cuántos ganó el underdog", axis=alt.Axis(format="%", tickCount=5),
              scale=alt.Scale(domain=[0, 0.45]))
    diagonal = alt.Chart(pd.DataFrame({"dijo": [0, 0.45], "paso": [0, 0.45]})).mark_line(
        color=COLOR["tinta_suave"], strokeDash=[2, 3], strokeWidth=1.5).encode(x=x, y=y)
    rotulo = alt.Chart(pd.DataFrame({"dijo": [0.40], "paso": [0.40], "t": ["predicción perfecta"]})).mark_text(
        angle=0, align="right", dx=-6, dy=-6, fontSize=12, fontStyle="italic", color=COLOR["tinta_suave"]).encode(x=x, y=y, text="t:N").properties(
        name=escritorio())
    lineas = alt.Chart(t).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=3).encode(
        x=x, y=y, color=alt.Color("serie:N", scale=escala, legend=leyenda_lineas()),
        strokeDash=alt.StrokeDash("serie:N", scale=trazo, legend=leyenda_lineas()),
        tooltip=[alt.Tooltip("serie:N", title=""), alt.Tooltip("dijo:Q", title="Dijo (promedio)", format=".1%"),
                 alt.Tooltip("paso:Q", title="Ganó el underdog", format=".1%"), alt.Tooltip("partidos:Q", title="Partidos")])
    return (diagonal + rotulo + lineas).properties(width="container", height=360)


def grafico_matriz(m: dict) -> alt.Chart:
    """La matriz de confusión del modelo en el test, con un umbral explícito (la tasa de victorias conocida antes del test)."""
    u = round(100 * m["umbral"])
    si, no = f"Dice {u}% o más", f"Dice menos de {u}%"
    gano, perdio = "Ganó", "No ganó"
    c = m["matriz"]
    filas = [(gano, si, c["vp"], "acierta"), (gano, no, c["fn"], "se le escapa"),
             (perdio, si, c["fp"], "falsa alarma"), (perdio, no, c["vn"], "acierta")]
    t = pd.DataFrame(filas, columns=["real", "dice", "n", "tipo"])
    t["de_la_fila"] = t["n"] / t.groupby("real")["n"].transform("sum")
    t["t"] = t.apply(lambda r: f"{r['n']:,}".replace(",", ".") + f" · {r['tipo']}", axis=1)
    t["t2"] = t.apply(lambda r: f"{round(100 * r['de_la_fila'])}% de los que " + ("ganaron" if r["real"] == gano else "no ganaron"), axis=1)
    t["acierto"] = t["tipo"] == "acierta"
    x = alt.X("dice:N", sort=[si, no], title=None, axis=alt.Axis(orient="top", labelFontSize=12, labelFontWeight="bold", labelAngle=0, labelLimit=260))
    y = alt.Y("real:N", sort=[gano, perdio], title=None, axis=alt.Axis(labelFontSize=12, labelFontWeight="bold"))
    celdas = alt.Chart(t).mark_rect(cornerRadius=6, stroke="white", strokeWidth=4).encode(
        x=x, y=y, color=alt.condition("datum.acierto", alt.value(COLOR["modelo_suave"]), alt.value("#f4d9c8")),
        tooltip=[alt.Tooltip("real:N", title="Lo que pasó"), alt.Tooltip("dice:N", title="Lo que dijo"), alt.Tooltip("n:Q", title="Partidos")])
    texto = alt.Chart(t).mark_text(fontSize=14, fontWeight="bold", dy=-9, color=COLOR["tinta"]).encode(x=x, y=y, text="t:N")
    texto2 = alt.Chart(t).mark_text(fontSize=11, dy=10, color=COLOR["tinta"]).encode(x=x, y=y, text="t2:N").properties(name=escritorio())
    return (celdas + texto + texto2).properties(width="container", height=190)


def grafico_roc(e: pd.DataFrame) -> alt.Chart:
    from sklearn.metrics import roc_auc_score, roc_curve
    test = e[es_test(e)]
    partes = []
    for serie, col in [("Modelo", "prob_modelo"), ("La cuota", "prob_mercado")]:
        fpr, tpr, _ = roc_curve(test["gano"], test[col])
        auc = roc_auc_score(test["gano"], test[col])
        partes.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "serie": f"{serie} (AUC {coma(auc, 3)})"}))
    t = pd.concat(partes, ignore_index=True)
    series = list(dict.fromkeys(t["serie"]))
    escala = alt.Scale(domain=series, range=[COLOR["modelo"], COLOR["mercado"]])
    trazo = alt.Scale(domain=series, range=[[1, 0], [7, 5]])
    x = alt.X("fpr:Q", title="Falsas alarmas (de los que no ganaron)", axis=alt.Axis(format="%", tickCount=5))
    y = alt.Y("tpr:Q", title="Aciertos (de los que ganaron)", axis=alt.Axis(format="%", tickCount=5))
    azar = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(color=COLOR["tinta_suave"], strokeDash=[2, 3]).encode(x=x, y=y)
    rot = alt.Chart(pd.DataFrame({"fpr": [0.74], "tpr": [0.6], "t": ["azar (AUC 0,5)"]})).mark_text(
        fontSize=11, fontStyle="italic", color=COLOR["tinta_suave"]).encode(x=x, y=y, text="t:N")
    curvas = alt.Chart(t).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
        x=x, y=y, color=alt.Color("serie:N", scale=escala, legend=leyenda_lineas(orient="bottom-right")),
        strokeDash=alt.StrokeDash("serie:N", scale=trazo, legend=leyenda_lineas(orient="bottom-right")), order="fpr:Q")
    return (azar + rot + curvas).properties(width=300, height=300)


# ----------------------------------------------------------------------------- 4 · ¿sirve para apostar?
def retorno(e: pd.DataFrame) -> pd.Series:
    """Lo que deja cada apuesta de $100 al underdog: ganancia neta si gana, −100 si no."""
    return APUESTA * (e["cuota"] * e["gano"] - 1)


def reglas_de_apuesta(e: pd.DataFrame) -> dict:
    """Las reglas de los competidores se fijan mirando SÓLO la validación (igual que el modelo); el test se mira después."""
    val = e[~es_test(e)]
    r = retorno(val)
    umbral = max(UMBRALES_SIN_FORMULA, key=lambda u: r[val["prob_modelo"] > u].mean())
    mejor = None
    for i, lo in enumerate(CORTES_RANGO):
        for hi in CORTES_RANGO[i + 1:]:
            m = (val["prob_mercado"] >= lo) & (val["prob_mercado"] < hi)
            if m.sum() >= MINIMO_APUESTAS_RANGO and (mejor is None or r[m].mean() > mejor[0]):
                mejor = (r[m].mean(), lo, hi)
    return {"umbral": umbral, "rango": (mejor[1], mejor[2])}


NOMBRES_ESTRATEGIAS = ["Modelo con la fórmula", "Modelo sin la fórmula", "Apostar a todos", "Dummy: rango de cuota", "Dummy: once mejor"]
COLORES_ESTRATEGIAS = [COLOR["modelo"], COLOR["modelo_suave"], COLOR["todos"], COLOR["mercado"], COLOR["once"]]
TRAZOS_ESTRATEGIAS = [[1, 0], [6, 4], [1, 0], [2, 3], [8, 3, 2, 3]]
ESTRATEGIA = alt.Scale(domain=NOMBRES_ESTRATEGIAS, range=COLORES_ESTRATEGIAS)


def estrategias(e: pd.DataFrame, reglas: dict) -> dict[str, tuple[pd.Series, str]]:
    """Cada estrategia: en qué partidos apuesta $100 al underdog, y su regla en palabras."""
    u, (lo, hi) = reglas["umbral"], reglas["rango"]
    return {
        "Modelo con la fórmula": (e["apuesta"] == 1, "probabilidad del modelo × cuota − 1 > 0"),
        "Modelo sin la fórmula": (e["prob_modelo"] > u, f"el modelo le da más de {round(100 * u)}% (umbral fijo elegido en validación)"),
        "Apostar a todos": (pd.Series(True, index=e.index), "siempre"),
        "Dummy: rango de cuota": ((e["prob_mercado"] >= lo) & (e["prob_mercado"] < hi),
                                  f"la cuota le da entre {round(100 * lo)}% y {round(100 * hi)}% (el mejor rango en validación)"),
        "Dummy: once mejor": (e["gap_xi"] > 0, "su once titular tiene mejor rating que el del favorito"),
    }


def sorteos_al_azar(e: pd.DataFrame, cuantas: int, rng: np.random.Generator) -> np.ndarray:
    """Retorno (%) de apostar al azar la misma cantidad de veces que el modelo, muchas veces."""
    r = retorno(e).to_numpy()
    return np.array([100 * r[rng.choice(len(r), cuantas, replace=False)].mean() / APUESTA for _ in range(SORTEOS_AZAR)])


def tabla_estrategias(e: pd.DataFrame, est: dict) -> tuple[list[dict], dict]:
    val, test = e[~es_test(e)], e[es_test(e)]
    rng = np.random.default_rng(0)
    filas, res = [], {}
    for nombre, (m, regla) in est.items():
        mv, mt = m[~es_test(e)], m[es_test(e)]
        rv, rt = retorno(val)[mv], retorno(test)[mt]
        boot = [rng.choice(rt.to_numpy(), len(rt)).mean() for _ in range(2000)]
        res[nombre] = {"val": rv.mean() / APUESTA, "test": rt.mean() / APUESTA, "n_val": len(rv), "n_test": len(rt),
                       "ic": (np.percentile(boot, 2.5) / APUESTA, np.percentile(boot, 97.5) / APUESTA)}
        filas.append({"estrategia": nombre, "apuesta cuando…": regla,
                      "validación 2010–15: apuestas": f"{len(rv):,}".replace(",", "."), "validación: retorno": signo_pct(rv.mean()),
                      "test 2015/16: apuestas": f"{len(rt):,}".replace(",", "."), "test: retorno": signo_pct(rt.mean()),
                      "test: IC 95%": f"{signo_pct(np.percentile(boot, 2.5), 0)} a {signo_pct(np.percentile(boot, 97.5), 0)}"})
    n_modelo_val, n_modelo_test = res["Modelo con la fórmula"]["n_val"], res["Modelo con la fórmula"]["n_test"]
    av, at = sorteos_al_azar(val, n_modelo_val, rng), sorteos_al_azar(test, n_modelo_test, rng)
    res["azar"] = {"val": float(np.median(av)) / 100, "test": float(np.median(at)) / 100,
                   "val_p95": float(np.percentile(av, 95)) / 100, "test_p95": float(np.percentile(at, 95)) / 100,
                   "test_supera": float((at >= 100 * res["Modelo con la fórmula"]["test"]).mean()),
                   "val_supera": float((av >= 100 * res["Modelo con la fórmula"]["val"]).mean())}
    filas.append({"estrategia": "Al azar", "apuesta cuando…": "sortea los partidos, tantas apuestas como el modelo (500 sorteos)",
                  "validación 2010–15: apuestas": f"{n_modelo_val:,}".replace(",", "."),
                  "validación: retorno": f"{signo_pct(np.median(av))} (95% de los sorteos: menos de {signo_pct(np.percentile(av, 95))})",
                  "test 2015/16: apuestas": f"{n_modelo_test:,}".replace(",", "."),
                  "test: retorno": f"{signo_pct(np.median(at))} (95% de los sorteos: menos de {signo_pct(np.percentile(at, 95))})", "test: IC 95%": ""})
    return filas, res


def grafico_formula(e: pd.DataFrame) -> alt.Chart:
    """Cómo decide la fórmula: compara lo que cree el modelo con lo que paga la cuota (test 2015/16)."""
    test = e[es_test(e)].assign(paga=lambda d: 1 / d["cuota"])
    test["decision"] = np.where(test["apuesta"] == 1, "Apuesta", "No apuesta")
    escala = alt.Scale(domain=["Apuesta", "No apuesta"], range=[COLOR["modelo"], "#c3c9d1"])
    dominio = [0, 0.45]
    x = alt.X("paga:Q", title="Lo que paga la cuota (1 ÷ cuota)", axis=alt.Axis(format="%", tickCount=5), scale=alt.Scale(domain=dominio))
    y = alt.Y("prob_modelo:Q", title="Lo que cree el modelo", axis=alt.Axis(format="%", tickCount=5),
              scale=alt.Scale(domain=dominio))
    puntos = alt.Chart(test).mark_circle(size=22, opacity=0.6, clip=True).encode(
        x=x, y=y, color=alt.Color("decision:N", scale=escala, title="El modelo",
                                  legend=alt.Legend(orient="top-left", labelFontSize=12, labelLimit=360, symbolSize=120,
                                                    fillColor="rgba(255,255,255,.92)", padding=6)),
        order=alt.Order("apuesta:Q"),
        tooltip=[alt.Tooltip("partido:N", title="Partido"), alt.Tooltip("cuota:Q", title="Cuota del underdog", format=".2f"),
                 alt.Tooltip("paga:Q", title="Paga como si tuviera", format=".1%"),
                 alt.Tooltip("prob_modelo:Q", title="El modelo cree", format=".1%"),
                 alt.Tooltip("valor:Q", title="Valor esperado", format="+.1%")])
    linea = alt.Chart(pd.DataFrame({"paga": dominio, "prob_modelo": dominio})).mark_line(
        color=COLOR["tinta"], strokeDash=[6, 4], strokeWidth=2).encode(x=x, y=y)
    rot_linea = alt.Chart(pd.DataFrame({"paga": [0.405], "prob_modelo": [0.405], "t": ["valor esperado 0"]})).mark_text(
        align="right", dx=-8, dy=-6, fontSize=12, fontWeight="bold", color=COLOR["tinta"]).encode(x=x, y=y, text="t:N")
    ejemplo = pd.DataFrame({"paga": [0.10], "prob_modelo": [0.15]})
    punto = alt.Chart(ejemplo).mark_point(size=260, shape="circle", strokeWidth=3, color=COLOR["tinta"], filled=False).encode(x=x, y=y).properties(
        name=escritorio())
    guia = alt.Chart(pd.DataFrame({"paga": [0.06], "prob_modelo": [0.285], "paga2": [0.097], "prob2": [0.158]})).mark_rule(
        color=COLOR["tinta"], strokeWidth=1.2).encode(x=x, y=y, x2="paga2:Q", y2="prob2:Q").properties(name=escritorio())
    texto = alt.Chart(pd.DataFrame({"paga": [0.008], "prob_modelo": [0.325],
                                    "t": ["Ejemplo: cuota 10 → paga como si tuviera 10%.\nSi el modelo cree 15%: 0,15 × 10 − 1 = +0,5\n→ +$50 esperados por cada $100: apuesta"]})).mark_text(
        align="left", baseline="middle", fontSize=12, lineBreak="\n", color=COLOR["tinta"]).encode(x=x, y=y, text="t:N").properties(name=escritorio())
    return (puntos + linea + rot_linea + guia + punto + texto).properties(width="container", height=380)


def grafico_modelo(e: pd.DataFrame, est: dict) -> alt.Chart:
    """Retorno acumulado en el test 2015/16, desde 0: lo ganado hasta cada fecha por cada $100 apostados hasta esa fecha.
    Se compara por cada $100 (y no en pesos totales) porque las estrategias apuestan cantidades de veces muy distintas."""
    test = e[es_test(e)].assign(**{nombre: m[es_test(e)] for nombre, (m, _) in est.items()})
    test = test.sort_values("fecha", kind="stable").reset_index(drop=True)
    r = retorno(test)
    fechas = sorted(test["fecha"].unique())
    inicio = (pd.to_datetime(fechas[0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    idx = test["fecha"].map({f: i for i, f in enumerate(fechas)}).to_numpy()

    def acumulado(elegidos: np.ndarray) -> np.ndarray:
        """Retorno acumulado por fecha; vacío hasta que la estrategia lleva MINIMO_APUESTAS_CURVA apuestas (antes es puro ruido)."""
        ganado = np.cumsum(np.bincount(idx[elegidos], weights=r.to_numpy()[elegidos], minlength=len(fechas)))
        apuestas = np.cumsum(np.bincount(idx[elegidos], minlength=len(fechas)))
        roi = np.full(len(fechas), np.nan)
        listo = apuestas >= MINIMO_APUESTAS_CURVA
        roi[listo] = 100 * ganado[listo] / (apuestas[listo] * APUESTA)
        return roi

    largo = pd.concat([pd.DataFrame({"fecha": fechas, "estrategia": nombre,
                                     "roi": acumulado(np.flatnonzero(test[nombre].to_numpy()))}) for nombre in est], ignore_index=True)
    largo = largo.dropna()
    # banda del azar: apostar a partidos sorteados, tantas veces como el modelo, 500 veces
    rng = np.random.default_rng(1)
    cuantas = int(test["Modelo con la fórmula"].sum())
    sorteos = np.array([acumulado(np.sort(rng.choice(len(test), cuantas, replace=False))) for _ in range(SORTEOS_AZAR)])
    with warnings.catch_warnings():   # las fechas antes de la apuesta 20 son todas vacías: se descartan abajo
        warnings.simplefilter("ignore", RuntimeWarning)
        banda = pd.DataFrame({"fecha": fechas, "bajo": np.nanpercentile(sorteos, 5, axis=0), "alto": np.nanpercentile(sorteos, 95, axis=0)})
    banda = banda[np.isfinite(sorteos).all(axis=0)]
    ultimo = pd.to_datetime(fechas[-1])
    titulo = "Retorno acumulado por cada $100 apostados (%)"
    x = alt.X("fecha:T", title=None, axis=alt.Axis(format="%b %Y", tickCount="month", grid=False, labelAngle=0),
              scale=alt.Scale(domain=[inicio, (ultimo + pd.Timedelta(days=110)).strftime("%Y-%m-%d")]))
    escala_y = alt.Scale(domain=[-45, 45], clamp=True)
    y = alt.Y("roi:Q", title=titulo, axis=alt.Axis(tickCount=8, format="+d"), scale=escala_y)
    area = alt.Chart(banda).mark_area(color=COLOR["azar"], opacity=0.75).encode(
        x=x, y=alt.Y("bajo:Q", title=titulo, scale=escala_y), y2="alto:Q",
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("bajo:Q", title="Azar: 5% peor (%)", format="+.1f"),
                 alt.Tooltip("alto:Q", title="Azar: 5% mejor (%)", format="+.1f")])
    cero = alt.Chart(pd.DataFrame({"roi": [0]})).mark_rule(color=COLOR["tinta_suave"], strokeWidth=1).encode(y=alt.Y("roi:Q", scale=escala_y))
    lineas = alt.Chart(largo).mark_line(interpolate="step-after", clip=True).encode(
        x=x, y=y, color=alt.Color("estrategia:N", scale=ESTRATEGIA, legend=None),
        strokeDash=alt.StrokeDash("estrategia:N", scale=alt.Scale(domain=NOMBRES_ESTRATEGIAS, range=TRAZOS_ESTRATEGIAS), legend=None),
        strokeWidth=alt.condition("datum.estrategia == 'Modelo con la fórmula'", alt.value(4), alt.value(2.2)),
        tooltip=[alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("estrategia:N", title="Estrategia"),
                 alt.Tooltip("roi:Q", title="Retorno acumulado (%)", format="+.1f")]).properties(name="lineas")
    # rótulo directo al final de cada línea (y de la banda), corridos para que no se pisen
    fin = largo[largo["fecha"] == fechas[-1]][["estrategia", "roi"]].copy()
    fin = pd.concat([fin, pd.DataFrame([{"estrategia": "Al azar (90% de los sorteos)", "roi": float(np.nanmedian(sorteos[:, -1]))},
                                        {"estrategia": "No apostar nunca", "roi": 0.0}])], ignore_index=True)
    fin = fin.sort_values("roi").reset_index(drop=True)
    pos = list(fin["roi"])
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + 6)
    fin["y"] = pos
    fin["fecha"] = fechas[-1]
    fin["t"] = fin.apply(lambda f: f"{f['estrategia']}: {signo_pct(f['roi'])}" if f["estrategia"] != "No apostar nunca" else "No apostar nunca: 0%", axis=1)
    colores = dict(zip(NOMBRES_ESTRATEGIAS, COLORES_ESTRATEGIAS)) | {"Al azar (90% de los sorteos)": "#8a94a1", "No apostar nunca": COLOR["tinta_suave"]}
    fin["color"] = fin["estrategia"].map(colores)
    rotulos = alt.Chart(fin).mark_text(align="left", dx=6, fontSize=12, fontWeight="bold").encode(
        x="fecha:T", y=alt.Y("y:Q", scale=escala_y), text="t:N", color=alt.Color("color:N", scale=None)).properties(name=escritorio())
    return (area + cero + lineas + rotulos).properties(width="container", height=400)


# ----------------------------------------------------------------------------- la copa
def grafico_explorar(ligas: list[str], reglas: dict) -> alt.Chart:
    """Ancho fijo de referencia: la página lo ajusta al ancho de la pantalla antes de dibujarlo (ver plantilla).
    Muestra sólo el test 2015/16, la temporada que el modelo final nunca vio."""
    datos = alt.UrlData("data/partidos.json")
    liga = alt.param(name="liga", value=None, bind=alt.binding_select(options=[None] + ligas, labels=["Todas las ligas"] + ligas, name="Liga"))
    zona = alt.selection_interval(name="zona", encodings=["x", "y"])
    filtro = "indexof(datum.conjunto, 'test') == 0 && (liga == null || datum.liga == liga)"
    x = alt.X("prob_mercado:Q", title="Probabilidad del underdog según la cuota", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.36]))
    y = alt.Y("valor:Q", title="Valor esperado de apostarle, según el modelo", axis=alt.Axis(format="+%"), scale=alt.Scale(domain=[-0.4, 0.3]))
    detalle = [alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"), alt.Tooltip("liga:N", title="Liga"),
               alt.Tooltip("partido:N", title="Partido"), alt.Tooltip("underdog:N", title="Underdog"),
               alt.Tooltip("cuota:Q", title="Cuota del underdog", format=".2f"),
               alt.Tooltip("prob_mercado:Q", title="Prob. según la cuota", format=".1%"),
               alt.Tooltip("prob_modelo:Q", title="Prob. según el modelo", format=".1%"),
               alt.Tooltip("decision:N", title="El modelo")]

    fondo = alt.Chart(datos).transform_filter(filtro).transform_filter("datum.apuesta == 0").mark_circle(
        size=14, opacity=0.45, color=COLOR["fondo_puntos"], clip=True).encode(x=x, y=y, tooltip=detalle)
    apuestas = alt.Chart(datos).transform_filter(filtro).transform_filter("datum.apuesta == 1").mark_circle(
        size=46, opacity=0.9, stroke="white", strokeWidth=0.6, clip=True).encode(
        x=x, y=y, tooltip=detalle, order=alt.Order("resultado:N", sort="descending"),  # los que ganaron se dibujan encima
        color=alt.Color("resultado:N", scale=RESULTADO, title="Apuestas del modelo",
                        legend=alt.Legend(orient="top-left", direction="horizontal", titleFontSize=12, labelFontSize=12)))
    cero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=COLOR["modelo"], strokeWidth=1.5, strokeDash=[6, 4]).encode(y="y:Q")
    rotulo = alt.Chart(pd.DataFrame({"y": [0], "t": ["arriba de esta línea, el modelo apuesta"]})).mark_text(
        align="left", x=6, dy=14, color=COLOR["modelo"], fontWeight="bold", fontSize=12).encode(y="y:Q", text="t:N")
    nube = (fondo + apuestas + cero + rotulo).add_params(liga, zona).properties(width=760, height=400)

    u, (lo, hi) = reglas["umbral"], reglas["rango"]
    plata = alt.Chart(datos).transform_filter(filtro).transform_filter(zona).transform_calculate(
        r=f"{APUESTA} * (datum.cuota * datum.gano - 1)",
        m_sin=f"datum.prob_modelo > {u} ? 1 : 0",
        m_rango=f"datum.prob_mercado >= {lo} && datum.prob_mercado < {hi} ? 1 : 0",
        m_once="datum.gap_xi > 0 ? 1 : 0",
    ).transform_calculate(
        r_formula="datum.r * datum.apuesta", r_sin="datum.r * datum.m_sin", r_rango="datum.r * datum.m_rango", r_once="datum.r * datum.m_once"
    ).transform_aggregate(
        partidos="count()", s_todos="sum(r)", s_formula="sum(r_formula)", n_formula="sum(apuesta)", s_sin="sum(r_sin)", n_sin="sum(m_sin)",
        s_rango="sum(r_rango)", n_rango="sum(m_rango)", s_once="sum(r_once)", n_once="sum(m_once)"
    ).transform_calculate(**{
        NOMBRES_ESTRATEGIAS[0]: "datum.n_formula > 0 ? datum.s_formula / datum.n_formula : 0",
        NOMBRES_ESTRATEGIAS[1]: "datum.n_sin > 0 ? datum.s_sin / datum.n_sin : 0",
        NOMBRES_ESTRATEGIAS[2]: "datum.s_todos / datum.partidos",
        NOMBRES_ESTRATEGIAS[3]: "datum.n_rango > 0 ? datum.s_rango / datum.n_rango : 0",
        NOMBRES_ESTRATEGIAS[4]: "datum.n_once > 0 ? datum.s_once / datum.n_once : 0",
    }).transform_fold(NOMBRES_ESTRATEGIAS, as_=["estrategia", "roi"])
    escala_roi = alt.Scale(domain=[-40, 40], clamp=True)
    barras = plata.mark_bar(height=22, cornerRadius=3).encode(
        y=alt.Y("estrategia:N", title=None, sort=NOMBRES_ESTRATEGIAS, axis=alt.Axis(labelFontSize=12, labelFontWeight="bold", labelLimit=220)),
        x=alt.X("roi:Q", title="Retorno por cada $100 apostados (%), en lo filtrado", scale=escala_roi),
        color=alt.Color("estrategia:N", scale=ESTRATEGIA, legend=None),
        tooltip=[alt.Tooltip("estrategia:N", title="Estrategia"), alt.Tooltip("roi:Q", title="Retorno %", format="+.1f"),
                 alt.Tooltip("partidos:Q", title="Partidos filtrados", format=",")])
    texto = plata.transform_filter("datum.roi >= 0").mark_text(align="left", dx=6, fontWeight="bold", fontSize=13).encode(
        y=alt.Y("estrategia:N", sort=NOMBRES_ESTRATEGIAS), x=alt.X("roi:Q", scale=escala_roi), text=alt.Text("roi:Q", format="+.1f"))
    texto_neg = plata.transform_filter("datum.roi < 0").mark_text(align="right", dx=-6, fontWeight="bold", fontSize=13).encode(
        y=alt.Y("estrategia:N", sort=NOMBRES_ESTRATEGIAS), x=alt.X("roi:Q", scale=escala_roi), text=alt.Text("roi:Q", format="+.1f"))
    resumen = alt.Chart(datos).transform_filter(filtro).transform_filter(zona).transform_aggregate(
        partidos="count()", apuestas="sum(apuesta)").transform_calculate(
        t="format(datum.partidos, ',') + ' partidos en la selección · el modelo apuesta en ' + format(datum.apuestas, ',')"
    ).mark_text(align="left", x=0, fontSize=13, fontWeight="bold", color=COLOR["tinta"]).encode(text="t:N").properties(width=760, height=20)
    cero_roi = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=COLOR["tinta_suave"]).encode(x="x:Q")
    return alt.vconcat(nube, resumen, (barras + texto + texto_neg + cero_roi).properties(width=760, height=150),
                       spacing=14).resolve_scale(color="independent")


def grafico_probar() -> alt.Chart:
    # fuentes con nombre y sin valores: la página les carga el caso que arma el usuario (view.data en JavaScript)
    caso = alt.NamedData(name="caso")
    escala = alt.Scale(domain=["Según la cuota", "Según el modelo"], range=[COLOR["mercado"], COLOR["modelo"]])
    barras = alt.Chart(caso).mark_bar(height=34).encode(
        y=alt.Y("quien:N", title=None, sort=["Según la cuota", "Según el modelo"]),
        x=alt.X("p:Q", title="Probabilidad de que gane el underdog", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.6])),
        color=alt.Color("quien:N", scale=escala, legend=None),
        tooltip=[alt.Tooltip("quien:N", title=""), alt.Tooltip("p:Q", title="Probabilidad", format=".1%")])
    texto = barras.mark_text(align="left", dx=6, fontWeight="bold", color="#222").encode(text=alt.Text("p:Q", format=".1%"))
    equilibrio = alt.NamedData(name="equilibrio")
    regla = alt.Chart(equilibrio).mark_rule(color=COLOR["todos"], strokeDash=[5, 3], strokeWidth=2).encode(x="p:Q")
    rotulo = alt.Chart(equilibrio).mark_text(align="left", dx=5, dy=-48, color=COLOR["todos"]).encode(
        x="p:Q", text=alt.value("lo que paga la cuota (1 / cuota)"))
    return (barras + texto + regla + rotulo).properties(width="container", height=120)


# ----------------------------------------------------------------------------- tablas de los «Cómo lo sabemos»
def tabla_html(registros: list[dict], columnas: list[str] | None = None, formatos: dict | None = None) -> str:
    formatos = formatos or {}
    columnas = columnas or list(registros[0])
    filas = []
    for r in registros:
        celdas = []
        for c in columnas:
            v = r.get(c)
            if c in formatos and v not in (None, "", "nan"):
                try:
                    v = formatos[c].format(float(v))
                except (TypeError, ValueError):
                    pass
            celdas.append(f"<td>{html.escape('' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))}</td>")
        filas.append("<tr>" + "".join(celdas) + "</tr>")
    cab = "".join(f"<th>{html.escape('' if c == 'fila' else c)}</th>" for c in columnas)
    return f"<div class='tbl'><table><tr>{cab}</tr>{''.join(filas)}</table></div>"


def embudo() -> str:
    ex = pd.read_parquet(AUDIT_DIR / "excluded_matches.parquet")["motivo"].value_counts()
    pasos = [("liga_sin_cuotas", "Se sacan las 2 ligas sin cuotas (Polonia y Suiza)"),
             ("alineacion_incompleta", "Se exige la alineación titular completa (22 jugadores)"),
             ("sin_cuotas_1x2", "Se exigen cuotas 1X2 completas de al menos una casa"),
             ("cuota_empate_incoherente", "Se sacan cuotas incoherentes (empate implícito > 40%)"),
             ("sin_underdog_identificable", "Se sacan partidos con las dos probabilidades idénticas"),
             ("partido_parejo", "Se exige un underdog claro: diferencia de probabilidad > 0,05")]
    quedan = 25_979
    filas = [{"paso": "Partidos en la fuente (European Soccer Database)", "se sacan": "", "quedan": f"{quedan:,}".replace(",", ".")}]
    for motivo, texto in pasos:
        n = int(ex.get(motivo, 0))
        quedan -= n
        filas.append({"paso": texto, "se sacan": f"{n:,}".replace(",", "."), "quedan": f"{quedan:,}".replace(",", ".")})
    return tabla_html(filas)


# ----------------------------------------------------------------------------- casos para «Probá el modelo»
def casos_ejemplo(df: pd.DataFrame) -> list[dict]:
    test = df[df["temporada"] == TEST_TEMPORADA].copy()
    test["valor"] = test["prob_modelo"] * test["cuota_underdog"] - 1
    sorpresas = test[test[TARGET_COLUMN] == 1].nsmallest(2, "prob_modelo")      # las dos sorpresas más grandes del test
    mas_valor = test.nlargest(2, "valor")                                        # las apuestas que el modelo más quería
    apuesta_ganada = test[(test["valor"] > 0) & (test[TARGET_COLUMN] == 1)].nlargest(1, "valor")
    elegidos = [(r, "la apuesta con más valor") for _, r in mas_valor.iterrows()]
    elegidos += [(r, "apuesta que salió") for _, r in apuesta_ganada.iterrows()]
    elegidos += [(r, "sorpresa que nadie vio") for _, r in sorpresas.iterrows()]
    elegidos += [(r, "") for _, r in test.sample(3, random_state=42).iterrows()]
    return [caso(r, etiqueta) for r, etiqueta in elegidos]


def caso(r: pd.Series, etiqueta: str = "") -> dict:
    """Un partido real, con todo lo que «Probá el modelo» necesita para cargarlo."""
    return {
        "etiqueta": etiqueta, "liga": LIGAS.get(r["liga"], r["liga"]), "fecha": r["fecha"].strftime("%d/%m/%Y"),
        "local": r["equipo_local"], "visitante": r["equipo_visitante"],
        "resultado": f"Terminó {r['goles_local']}–{r['goles_visitante']}",
        "cuotas": [float(r["odds_home"]), float(r["odds_draw"]), float(r["odds_away"])],
        "gaps": {c: (None if pd.isna(r[c]) else round(float(r[c]), 4)) for c in NOMBRES_GAPS},
    }


def partidos_del_test(df: pd.DataFrame) -> list[dict]:
    """Todos los partidos de la temporada de test (los que el modelo final nunca vio), para elegir por liga y equipo."""
    test = df[df["temporada"] == TEST_TEMPORADA].sort_values(["fecha", "match_id"])
    return [caso(r) for _, r in test.iterrows()]


# ----------------------------------------------------------------------------- página
def spec(chart: alt.Chart) -> str:
    return chart.configure_view(stroke=None).configure_axis(labelFontSize=12, titleFontSize=12, titleColor="#444",
                                                            gridColor="#eceef1").to_json(indent=None)


def piezas_de_la_historia(explorar: pd.DataFrame, caso_problema: dict, t_datos: pd.DataFrame, tasa_media: float,
                          t_hallazgo: pd.DataFrame) -> tuple[dict, dict]:
    """Los gráficos del tallo y de la exploración, y los números que los textos citan.
    Sale de las predicciones fuera de muestra (lo mismo que se publica en docs/data/partidos.json) y de dos tablitas resumen
    de Silver, así que con los mismos datos da siempre lo mismo."""
    pct = lambda x, d=1: f"{100 * x:.{d}f}".replace(".", ",") + "%"  # noqa: E731
    sig = lambda x, d=1: signo_pct(100 * x, d)  # noqa: E731
    miles_ = lambda n: f"{n:,}".replace(",", ".")  # noqa: E731

    g_problema, p = grafico_problema(caso_problema)
    g_datos, n_datos = grafico_datos(t_datos, tasa_media)
    g_hallazgo, n_hallazgo = grafico_hallazgo(t_hallazgo)
    met = metricas_prediccion(explorar)
    reglas = reglas_de_apuesta(explorar)
    est = estrategias(explorar, reglas)
    filas, res = tabla_estrategias(explorar, est)
    ligas = sorted(explorar.loc[es_test(explorar), "liga"].unique())
    graficos = {"g_problema": spec(g_problema), "g_datos": spec(g_datos), "g_hallazgo": spec(g_hallazgo),
                "g_calibracion": spec(grafico_calibracion(explorar)), "g_matriz": spec(grafico_matriz(met)),
                "g_roc": spec(grafico_roc(explorar)), "g_formula": spec(grafico_formula(explorar)),
                "g_modelo": spec(grafico_modelo(explorar, est)), "g_explorar": spec(grafico_explorar(ligas, reglas))}
    m = met["matriz"]
    r = {k: res[k] for k in NOMBRES_ESTRATEGIAS}
    historia = {
        "P_LOCAL": html.escape(p["local"]), "P_VISIT": html.escape(p["visitante"]), "P_FAV": html.escape(p["favorito"]),
        "P_UND": html.escape(p["underdog"]), "P_CUOTAS": " / ".join(coma(c, 2) for c in p["cuotas"]),
        "P_CUOTA_UND": coma(p["cuota_underdog"], 2), "P_COBRO": coma(APUESTA * p["cuota_underdog"], 0),
        "P_GANANCIA": coma(APUESTA * (p["cuota_underdog"] - 1), 0), "P_PROB_UND": pct(p["p_underdog"]), "P_PROB_FAV": pct(p["p_favorito"]),
        "P_SUMA": pct(p["suma"]), "P_MARGEN": pct(p["suma"] - 1),
        "DATOS_ALTO": pct(n_datos["alto"], 0), "DATOS_BAJO": pct(n_datos["bajo"], 0),
        "XI_BAJO": pct(n_hallazgo["bajo"], 0), "XI_ALTO": pct(n_hallazgo["alto"], 0),
        "T_PREDICCION": tabla_html(met["filas"], formatos={"AUC: ¿ordena? (más es mejor)": "{:.3f}", "log-loss (menos es mejor)": "{:.4f}",
                                                          "Brier (menos es mejor)": "{:.4f}"}).replace(".", ","),
        "UMBRAL_MATRIZ": pct(met["umbral"], 0), "M_VP": miles_(m["vp"]), "M_FN": miles_(m["fn"]), "M_FP": miles_(m["fp"]), "M_VN": miles_(m["vn"]),
        "M_RECALL": pct(m["vp"] / (m["vp"] + m["fn"]), 0), "M_PRECISION": pct(m["vp"] / (m["vp"] + m["fp"]), 0),
        "TASA_TEST": pct(met["tasa_test"]), "TASA_VAL": pct(met["tasa_val"]), "N_TEST": miles_(met["n_test"]), "N_VAL": miles_(met["n_val"]),
        "T_ESTRATEGIAS": tabla_html(filas),
        "UMBRAL_SIN": pct(reglas["umbral"], 0), "RANGO_LO": pct(reglas["rango"][0], 0), "RANGO_HI": pct(reglas["rango"][1], 0),
        "R_FORMULA_VAL": sig(r["Modelo con la fórmula"]["val"]), "R_FORMULA_TEST": sig(r["Modelo con la fórmula"]["test"]),
        "R_SIN_VAL": sig(r["Modelo sin la fórmula"]["val"]), "R_SIN_TEST": sig(r["Modelo sin la fórmula"]["test"]),
        "R_TODOS_VAL": sig(r["Apostar a todos"]["val"]), "R_TODOS_TEST": sig(r["Apostar a todos"]["test"]),
        "R_RANGO_VAL": sig(r["Dummy: rango de cuota"]["val"]), "R_RANGO_TEST": sig(r["Dummy: rango de cuota"]["test"]),
        "R_ONCE_VAL": sig(r["Dummy: once mejor"]["val"]), "R_ONCE_TEST": sig(r["Dummy: once mejor"]["test"]),
        "N_APUESTAS_TEST": miles_(r["Modelo con la fórmula"]["n_test"]), "N_APUESTAS_VAL": miles_(r["Modelo con la fórmula"]["n_val"]),
        "AZAR_TEST_P95": sig(res["azar"]["test_p95"]), "AZAR_VAL_P95": sig(res["azar"]["val_p95"]),
        "AZAR_TEST_SUPERA": pct(res["azar"]["test_supera"], 0), "AZAR_VAL_SUPERA": pct(res["azar"]["val_supera"], 0),
        "N_EXPLORAR": miles_(int(es_test(explorar).sum())),
    }
    return graficos, historia


def construir() -> None:
    df, ficha, resultados = cargar()
    dif = verificar_contra_el_pipeline(df)
    assert dif < 1e-9, f"La fórmula de la app no coincide con el modelo del notebook ({dif})"

    afuera = fuera_de_muestra(df, resultados)
    explorar = partidos_para_explorar(afuera)
    (SALIDA / "data").mkdir(parents=True, exist_ok=True)
    explorar.to_json(SALIDA / "data" / "partidos.json", orient="records", force_ascii=False)
    (SALIDA / "data" / "partidos_test.json").write_text(json.dumps(partidos_del_test(df), ensure_ascii=False), encoding="utf-8")
    (SALIDA / ".nojekyll").write_text("", encoding="utf-8")

    graficos, historia = piezas_de_la_historia(explorar, casos_ejemplo(df)[0], tabla_datos(df), float(df[TARGET_COLUMN].mean()),
                                               tabla_hallazgo(df))
    graficos["g_probar"] = spec(grafico_probar())

    val, test = ficha["validacion"], ficha["test"]
    pct = lambda x, d=1: f"{100 * x:.{d}f}".replace(".", ",") + "%"  # noqa: E731
    signo = lambda x: f"{100 * x:+.1f}".replace(".", ",").replace("-", "−") + "%"  # noqa: E731  (retornos: siempre con signo)
    num = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
    fecha_datos = json.loads(SILVER_MANIFEST_PATH.read_text(encoding="utf-8")).get("built_at_utc", "")[:10]

    rangos = {c: {"min": v["min_entrenamiento"], "max": v["max_entrenamiento"], "nombre": NOMBRES_GAPS[c]}
              for c, v in ficha["numericas"].items()}
    entrenamiento = df[df["temporada"] != TEST_TEMPORADA]
    rangos.update({c: {"min": float(entrenamiento[c].min()), "max": float(entrenamiento[c].max()), "nombre": c}
                   for c in ficha["probabilidades"]})
    contexto = {
        "TASA": pct(df[TARGET_COLUMN].mean()), "N": f"{len(df):,}".replace(",", "."),
"AUC_TEST_2": num(test["auc"], 2), "AUC_MERCADO_2": num(test["auc_mercado"], 2),
        "AUC_TEST": num(test["auc"]), "AUC_MERCADO": num(test["auc_mercado"]), "AUC_VAL": num(val["auc_media"]),
        "AUC_VAL_SD": num(val["auc_desvio"]), "ROI_TODOS": signo(val["roi_todos"]), "ROI_MODELO": signo(val["roi"]),
        "ROI_TEST": pct(test["roi"]), "ROI_TODOS_TEST": pct(test["roi_todos"]), "APUESTAS_TEST": str(test["apuestas"]),
        "FECHA_DATOS": fecha_datos, "EMBUDO": embudo(), "FUGA": ", ".join(f"<code>{c}</code>" for c in LEAKAGE_COLUMNS),
        "T_COMPARACION": tabla_html(resultados["comparacion"], formatos={"AUC media": "{:.3f}", "AUC desvío": "{:.3f}", "AUC train": "{:.3f}",
                                                                         "log-loss media": "{:.4f}", "Δ AUC % vs logística": "{:+.2f}%"}),
        "T_GRUPOS": tabla_html(resultados["grupos"], formatos={"AUC media": "{:.3f}", "AUC desvío": "{:.3f}",
                                                               "Δ AUC vs prob. sola": "{:+.4f}", "desvío de Δ": "{:.4f}"}),
        "T_PLATA": tabla_html(resultados["plata"], formatos={"% de partidos": "{:.1f}%", "ROI %": "{:+.2f}%"}),
        "T_PEDAZOS": tabla_html(resultados["plata_por_pedazo"], formatos={k: "{:+.1f}%" for k in resultados["plata_por_pedazo"][0] if k not in ("fila", "casa")}),
        "T_TEST": tabla_html(resultados["test"], formatos={"AUC": "{:.3f}", "log-loss": "{:.4f}", "ROI %": "{:+.2f}%"}),
        "T_COEF": tabla_html(resultados["coeficientes"], formatos={"coeficiente": "{:+.3f}"}),
        "C": str(ficha["C"]), "INTERCEPTO": num(ficha["intercepto"]),
        "COEF_PROB": num(ficha["probabilidades"]["prob_no_favorito"]["coef"]),
        "GRAFICOS": json.dumps(graficos, ensure_ascii=False),
        "MODELO": json.dumps({"intercepto": ficha["intercepto"], "probabilidades": ficha["probabilidades"],
                              "numericas": ficha["numericas"], "rangos": rangos}, ensure_ascii=False),
        "CASOS": json.dumps(casos_ejemplo(df), ensure_ascii=False),
        "VEGA": VEGA, "VEGA_LITE": VEGA_LITE, "VEGA_EMBED": VEGA_EMBED,
    } | historia
    pagina = (Path(__file__).parent / "plantilla.html").read_text(encoding="utf-8")
    for clave, valor in contexto.items():
        pagina = pagina.replace("{{" + clave + "}}", valor)
    assert "{{" not in pagina, "quedó un marcador sin reemplazar en la plantilla"
    (SALIDA / "index.html").write_text(pagina, encoding="utf-8")
    print(f"App generada en {SALIDA / 'index.html'} ({len(pagina) / 1024:.0f} KB) y "
          f"{SALIDA / 'data' / 'partidos.json'} ({len(explorar):,} partidos). Fórmula = notebook (dif. {dif:.1e}).")


if __name__ == "__main__":
    construir()
