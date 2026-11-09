"""Modelo de datos: perfiles de CPU, tráfico (Mbps) y latencia (ms) de los enlaces y su PESO para Dijkstra.

Cada arista guarda explícitamente:
    trafico  tráfico medido en el enlace (Mbps)
    latencia retardo del enlace (ms)                      variable prioritaria
    activa   disponibilidad del enlace (Activa / Interrumpida / Inactiva)
    peso     costo que usa Dijkstra (adimensional)

Como el tráfico (Mbps) y la latencia (ms) tienen unidades distintas, cada uno se
normaliza dividiéndolo por el valor máximo de su rango de generación en
condiciones normales: 140 Mbps (límite superior del troncal) y 15 ms (límite
superior del wifi). Así ambos quedan en [0, 1] y valores mayores que 1 indican
sobrecarga (por ejemplo, tras una falla):

    peso(u,v) = 0,30·(T/140 Mbps) + 0,70·(L/15 ms) + 0,2·A(u,v)

donde A(u,v) ∈ {0, 1, 2} es el número de extremos del enlace en estado Anómalo.
Un equipo Anómalo sigue siendo transitable: solo encarece sus enlaces. Como
T ≥ 1 Mbps y L > 0, el peso siempre es positivo, condición de Dijkstra.
Los equipos Bloqueados y los enlaces inactivos o interrumpidos se excluyen del
grafo de búsqueda.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

TIPOS = ("Servidor", "PC", "Portátil", "Impresora", "IoT")

# Uso de CPU esperado por tipo: (media, desviación) en %
PERFIL_CPU = {
    "Servidor": (55.0, 10.0),
    "PC": (40.0, 12.0),
    "Portátil": (35.0, 12.0),
    "Impresora": (18.0, 7.0),
    "IoT": (25.0, 9.0),
}

# Rango de tráfico medido (Mbps) y latencia (ms) por tipo de enlace
ENLACES = {
    "troncal":  {"trafico": (60.0, 140.0), "latencia": (0.5, 2.0)},
    "cableado": {"trafico": (10.0, 70.0),  "latencia": (1.0, 5.0)},
    "wifi":     {"trafico": (5.0, 45.0),   "latencia": (4.0, 15.0)},
}
TRAFICO_MAXIMO = 300.0          # tope del tráfico de un enlace (Mbps)

# Peso de las aristas para Dijkstra (normalización y coeficientes)
TRAFICO_REF = 140.0             # Mbps: máximo del rango de generación (troncal)
LATENCIA_REF = 15.0             # ms: máximo del rango de generación (wifi)
ALFA_TRAFICO = 0.30             # 30 % tráfico
BETA_LATENCIA = 0.70            # 70 % latencia (variable prioritaria)
GAMMA_ANOMALO = 0.20            # penalización por cada extremo Anómalo (sigue siendo transitable)

# Clasificación del tráfico de un enlace
TRAFICO_MEDIO = 30.0            # ≥ 30 Mbps: tráfico medio
TRAFICO_ALTO = 60.0             # ≥ 60 Mbps: enlace congestionado

# Eventos de comunicación: P(ERR) = 0,02 + 0,05·máx(d − 1, 0)
PROB_ERR_BASE = 0.02
PROB_ERR_PENDIENTE = 0.05
PROB_ERR_SIN_SERVICIO = 0.85
TRAFICO_NORMAL = 45.0           # tráfico medio esperado en los enlaces de un equipo (Mbps)
DISPERSION_TRAFICO = 10.0       # Mbps por unidad de desviación


def desviacion(g: nx.Graph, nodo: str) -> float:
    """d_i = máx(z_CPU, (T̄ − 45)/10, 0): carga del equipo y tráfico medio (Mbps) de sus enlaces."""
    d = g.nodes[nodo]
    media, sigma = PERFIL_CPU[d["tipo"]]
    z_cpu = (d["cpu"] - media) / sigma
    trafico = [a["trafico"] for _, _, a in g.edges(nodo, data=True) if a.get("activa", True)]
    z_red = (float(np.mean(trafico)) - TRAFICO_NORMAL) / DISPERSION_TRAFICO if trafico else 0.0
    return max(z_cpu, z_red, 0.0)


def prob_error(g: nx.Graph, nodo: str) -> float:
    """Probabilidad de que un intervalo de comunicación termine en ERR."""
    if g.nodes[nodo].get("sin_servicio"):
        return PROB_ERR_SIN_SERVICIO
    d = desviacion(g, nodo)
    return float(np.clip(PROB_ERR_BASE + PROB_ERR_PENDIENTE * max(d - 1.0, 0.0), PROB_ERR_BASE, 0.85))


def calcular_peso(trafico: float, latencia: float, extremos_anomalos: int = 0) -> float:
    """peso = 0,30·T/140 + 0,70·L/15 + 0,2·A  (positivo y adimensional)."""
    return (ALFA_TRAFICO * trafico / TRAFICO_REF + BETA_LATENCIA * latencia / LATENCIA_REF
            + GAMMA_ANOMALO * extremos_anomalos)


def extremos_anomalos(g: nx.Graph, u: str, v: str) -> int:
    return sum(g.nodes[x].get("estado_afd") == "Anómalo" for x in (u, v))


def actualizar_pesos(g: nx.Graph) -> None:
    """Recalcula el atributo 'peso' de todas las aristas con las condiciones actuales de la red."""
    for u, v, a in g.edges(data=True):
        a["peso"] = round(calcular_peso(a["trafico"], a["latencia"], extremos_anomalos(g, u, v)), 4)


def normalizados(trafico: float, latencia: float) -> tuple[float, float]:
    """Tráfico y latencia normalizados (T/140, L/15)."""
    return trafico / TRAFICO_REF, latencia / LATENCIA_REF


def costo_real(g: nx.Graph, u: str, v: str) -> float:
    """Costo de atravesar el enlace u–v: su atributo 'peso' (el mismo que usa Dijkstra)."""
    return g.edges[u, v]["peso"]


NOMBRES_CARACTERISTICAS = (
    "trafico/140", "(trafico/140)²", "latencia/15", "inalambrico",
    "extremos_anomalos", "cpu_llegada/100",
    "es_Servidor", "es_PC", "es_Portátil", "es_Impresora", "es_IoT",
)


def caracteristicas_arista(g: nx.Graph, u: str, v: str) -> np.ndarray:
    """Vector φ(u→v) que recibe la red neuronal para estimar el costo del enlace (no recibe el peso)."""
    a, dv = g.edges[u, v], g.nodes[v]
    tipo = np.zeros(len(TIPOS))
    tipo[TIPOS.index(dv["tipo"])] = 1.0
    t, lat = normalizados(a["trafico"], a["latencia"])
    return np.concatenate([[t, t * t, lat, float(a["clase"] == "wifi"),
                            float(extremos_anomalos(g, u, v)), dv["cpu"] / 100.0], tipo])
