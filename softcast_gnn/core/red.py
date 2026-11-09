"""Red fija de SoftCast: los 18 equipos son los nodos y los 29 enlaces de comunicación las aristas.

Cada sesión parte de una configuración aleatoria (reproducible con la semilla):
dos equipos Bloqueados y entre uno y tres enlaces interrumpidos.

G = (V, E, w): grafo no dirigido y ponderado. Cada arista guarda su tráfico
(Mbps), su latencia (ms), su disponibilidad (activa) y su peso, que combina
ambas variables normalizadas (ver datos.py) y es el atributo que usa Dijkstra.
Cada equipo tiene al menos tres enlaces y los equipos estratégicos cuatro; la
red es biconexa: el bloqueo de un equipo o de un enlace no la desconecta.
La topología es fija; el tráfico, la CPU, las latencias y los eventos son aleatorios.
"""

from __future__ import annotations

import copy
import zlib
from collections import deque

import networkx as nx
import numpy as np

from . import automata as au
from . import datos as dt

AFD = au.afd_dispositivo()

# --------------------------------------------------------------------------
# 1. Inventario y topología fijos
# --------------------------------------------------------------------------
SERVIDORES = ["S1", "S2"]
INVENTARIO = {
    "S1": ("Servidor", "Servidor de aplicaciones y archivos", "Administrativa"),
    "S2": ("Servidor", "Servidor de bases de datos y respaldo", "Operativa"),
    "PC1": ("PC", "Computador de escritorio", "Administrativa"),
    "PC2": ("PC", "Computador de escritorio", "Administrativa"),
    "PC3": ("PC", "Computador de escritorio", "Administrativa"),
    "PC4": ("PC", "Computador de escritorio", "Operativa"),
    "PC5": ("PC", "Computador de escritorio", "Operativa"),
    "PT1": ("Portátil", "Computador portátil (wifi)", "Administrativa"),
    "PT2": ("Portátil", "Computador portátil (wifi)", "Administrativa"),
    "PT3": ("Portátil", "Computador portátil (wifi)", "Operativa"),
    "PT4": ("Portátil", "Computador portátil (wifi)", "Operativa"),
    "PT5": ("Portátil", "Computador portátil (wifi)", "Administrativa"),
    "IM1": ("Impresora", "Impresora de red", "Administrativa"),
    "IM2": ("Impresora", "Impresora de red", "Operativa"),
    "IoT1": ("IoT", "Cámara de seguridad", "Administrativa"),
    "IoT2": ("IoT", "Sensor de temperatura", "Administrativa"),
    "IoT3": ("IoT", "Control de acceso", "Operativa"),
    "IoT4": ("IoT", "Cámara de seguridad", "Operativa"),
}

# Anillo de 16 equipos (sentido horario, desde la cámara IoT1 en la parte superior)
ANILLO = ("IoT1", "PC4", "PT3", "IM2", "IoT3", "PT4", "IoT4", "PC5",
          "PT5", "PC3", "IoT2", "PT2", "PC2", "IM1", "PT1", "PC1")
# Troncal entre servidores, seis acometidas desde los servidores (tres por servidor)
# y seis enlaces transversales que dan rutas alternativas entre los sectores
ENLACES_EXTRA = (("S1", "S2"),
                 ("S1", "PC1"), ("S1", "PC3"), ("S1", "IM1"),
                 ("S2", "PC4"), ("S2", "PC5"), ("S2", "IM2"),
                 ("PT1", "PC2"), ("PT3", "IoT3"), ("PT2", "PT4"),
                 ("IoT1", "PT1"), ("PT5", "IoT2"), ("PT5", "IoT4"))


def clase_enlace(u: str, v: str) -> str:
    """troncal entre servidores; wifi si participa un portátil o un IoT; si no, cableado."""
    tipos = {INVENTARIO[u][0], INVENTARIO[v][0]}
    if tipos == {"Servidor"}:
        return "troncal"
    return "wifi" if tipos & {"Portátil", "IoT"} else "cableado"


ENLACES_FIJOS = [(u, v, clase_enlace(u, v)) for u, v in
                 [(ANILLO[i], ANILLO[(i + 1) % len(ANILLO)]) for i in range(len(ANILLO))]
                 + list(ENLACES_EXTRA)]

ORDEN = list(INVENTARIO)


def ordenar_nodos(nodos) -> list[str]:
    return sorted(nodos, key=ORDEN.index)


def _rng(*partes) -> np.random.Generator:
    """Generador determinista a partir de varias claves (reproducibilidad)."""
    return np.random.default_rng(zlib.crc32("|".join(str(p) for p in partes).encode()))


def _fijar_trafico(a: dict, trafico: float) -> None:
    """Tráfico medido en Mbps (1 decimal); el peso se recalcula con actualizar_pesos."""
    a["trafico"] = round(float(np.clip(trafico, 1.0, dt.TRAFICO_MAXIMO)), 1)


def construir_topologia(semilla: int) -> nx.Graph:
    """Grafo fijo de 18 nodos y 29 aristas con tráfico (Mbps), latencia (ms) y peso."""
    rng = _rng("enlaces", semilla)
    g = nx.Graph(semilla=semilla)
    for v in ORDEN:
        tipo, rol, area = INVENTARIO[v]
        g.add_node(v, tipo=tipo, rol=rol, area=area)
    for u, v, clase in ENLACES_FIJOS:
        p = dt.ENLACES[clase]
        a = {"clase": clase, "activa": True, "latencia": round(float(rng.uniform(*p["latencia"])), 2)}
        _fijar_trafico(a, rng.uniform(*p["trafico"]))
        g.add_edge(u, v, **a)
    dt.actualizar_pesos(g)
    return g


# --------------------------------------------------------------------------
# 2. Métricas aleatorias y fallas
# --------------------------------------------------------------------------
def inyectar_falla(g: nx.Graph, nodo: str, rng: np.random.Generator) -> None:
    """Satura el equipo: CPU alta y +30 a +50 Mbps de tráfico en cada uno de sus caminos."""
    g.nodes[nodo]["cpu"] = float(rng.uniform(90, 99.5))
    g.nodes[nodo]["falla_inyectada"] = True
    for v in g.neighbors(nodo):
        a = g.edges[nodo, v]
        _fijar_trafico(a, a["trafico"] + rng.uniform(30.0, 50.0))


def generar_red(semilla: int = 5, proporcion_fallas: float = 0.12) -> nx.Graph:
    """Red fija con CPU y tráfico aleatorios y fallas iniciales en equipos que no son servidores."""
    g = construir_topologia(semilla)
    rng = _rng("metricas", semilla)
    for v in g.nodes:
        media, sigma = dt.PERFIL_CPU[g.nodes[v]["tipo"]]
        g.nodes[v].update(cpu=float(np.clip(rng.normal(media, sigma), 1, 100)), falla_inyectada=False)
    candidatos = [v for v in ORDEN if g.nodes[v]["tipo"] != "Servidor"]
    k = round(proporcion_fallas * len(candidatos)) if proporcion_fallas > 0 else 0
    for nodo in rng.choice(candidatos, size=k, replace=False) if k else []:
        inyectar_falla(g, str(nodo), rng)
    dt.actualizar_pesos(g)
    return g


# Configuración aleatoria de cada sesión: dos equipos bloqueados y enlaces interrumpidos
N_BLOQUEADOS_INICIO = 2


def configuracion_inicial(semilla: int) -> dict:
    """Elige al azar (reproducible con la semilla) los equipos bloqueados y los enlaces interrumpidos.

    * Exactamente dos equipos distintos (rng.choice sin reemplazo); no se bloquean los dos
      servidores a la vez, porque toda la red quedaría sin servicio.
    * Entre 1 y 3 enlaces interrumpidos, elegidos entre los que no tocan a un equipo
      bloqueado y solo si los equipos no bloqueados siguen conectados (integridad de la red).
    """
    rng = _rng("configuracion", semilla)
    while True:
        bloqueados = [str(v) for v in rng.choice(ORDEN, size=N_BLOQUEADOS_INICIO, replace=False)]
        if set(bloqueados) != set(SERVIDORES):
            break
    topo = construir_topologia(semilla)
    h = topo.subgraph([v for v in ORDEN if v not in bloqueados]).copy()
    candidatos = sorted((u, v) for u, v in h.edges)
    objetivo = int(rng.integers(1, 4))                 # 1, 2 o 3 enlaces interrumpidos
    interrumpidos = []
    for i in rng.permutation(len(candidatos)):
        if len(interrumpidos) == objetivo:
            break
        u, v = candidatos[int(i)]
        h.remove_edge(u, v)
        if nx.is_connected(h):                         # integridad: la red activa sigue conectada
            interrumpidos.append((u, v))
        else:
            h.add_edge(u, v)
    return {"semilla": semilla, "bloqueados": ordenar_nodos(bloqueados), "interrumpidos": interrumpidos}


# --------------------------------------------------------------------------
# 3. Recorrido BFS con evaluación del AFD
# --------------------------------------------------------------------------
def _evaluar_nodo(g: nx.Graph, v: str, habilitar: set[str]) -> None:
    """Genera la cadena de eventos del equipo y la evalúa con el AFD.

    Solo pueden quedar bloqueados por el AFD (dos ERR seguidos) los equipos en los
    que el usuario inyectó una falla o que quedaron sin servicio; para los demás la
    cadena se vuelve a sortear hasta que no contenga dos ERR seguidos. Así, en la
    configuración inicial hay exactamente dos equipos bloqueados y el resto queda
    Normal o Anómalo según su carga.
    """
    d = g.nodes[v]
    p = dt.prob_error(g, v)
    libre = d.get("sin_servicio") or v in g.graph.get("auto_bloqueo", set())
    for intento in range(40):
        rng = _rng("eventos", g.graph["semilla"], v, round(p, 2), d.get("bloqueo_manual", False), intento)
        cadena = au.generar_eventos(rng, p, bloqueo_manual=d.get("bloqueo_manual", False),
                                    habilitar=v in habilitar)
        if libre or d.get("bloqueo_manual") or AFD.estado_final(cadena) != au.BLOQUEADO:
            break
    else:                                    # caso extremo: se rompe la pareja ERR ERR
        for i in range(1, len(cadena)):
            if cadena[i] == cadena[i - 1] == au.ERR:
                cadena[i] = au.OK
    n_obs = sum(s in (au.OK, au.ERR) for s in cadena)
    d.update(cadena=cadena, prob_error=round(p, 3), estado_afd=AFD.estado_final(cadena),
             tasa_fallas=cadena.count(au.ERR) / max(1, n_obs),
             traza_estados=[AFD.inicial] + [t[2] for t in AFD.recorrer(cadena)])


def recorrer_red(g: nx.Graph, habilitar: set[str] = frozenset()) -> list[dict]:
    """ALGORITMO DE RECORRIDO: búsqueda en anchura (BFS) desde los servidores.

    Cada nodo visitado genera su cadena de eventos y el AFD fija su estado.
    Si queda Bloqueado no se exploran sus vecinos. Los nodos no alcanzados
    quedan sin servicio. Complejidad O(|V| + |E|).
    """
    for v in g.nodes:
        g.nodes[v].update(visitado=False, nivel=None, padre=None, sin_servicio=False)
    cola = deque()
    for s in SERVIDORES:
        if not g.nodes[s].get("bloqueo_activo"):
            g.nodes[s].update(visitado=True, nivel=0)
            cola.append(s)
    orden = []
    while cola:
        u = cola.popleft()
        _evaluar_nodo(g, u, habilitar)
        du = g.nodes[u]
        expande = du["estado_afd"] != au.BLOQUEADO
        orden.append({"paso": len(orden) + 1, "nodo": u, "nivel": du["nivel"], "padre": du["padre"],
                      "estado": du["estado_afd"], "expande": expande})
        if not expande:
            continue
        for v in ordenar_nodos(g.neighbors(u)):
            dv = g.nodes[v]
            if dv["visitado"] or dv.get("bloqueo_activo") or not g.edges[u, v]["activa"]:
                continue
            dv.update(visitado=True, nivel=du["nivel"] + 1, padre=u)
            cola.append(v)
    for v in ORDEN:
        dv = g.nodes[v]
        if dv["visitado"]:
            continue
        if not dv.get("bloqueo_activo"):
            dv["sin_servicio"] = True
        _evaluar_nodo(g, v, habilitar)
        orden.append({"paso": None, "nodo": v, "nivel": None, "padre": None,
                      "estado": dv["estado_afd"], "expande": False})
    return orden


# --------------------------------------------------------------------------
# 4. Simulación de escenarios
# --------------------------------------------------------------------------
def _desactivar(g: nx.Graph, nodo: str) -> None:
    for v in g.neighbors(nodo):
        g.edges[nodo, v]["activa"] = False


def simular(g_cruda: nx.Graph, nodos_bloqueados=(), enlaces_bloqueados=(), fallas=(),
            habilitar=(), bloqueo_inicial=(), interrupcion_inicial=()) -> nx.Graph:
    """Aplica un escenario sobre una copia de la red y la evalúa.

    bloqueo_inicial / interrupcion_inicial: configuración aleatoria de la sesión
    (dos equipos bloqueados y enlaces interrumpidos). nodos_bloqueados,
    enlaces_bloqueados, fallas y habilitar: cambios manuales del usuario.

    Orden: fallas → interrupción de enlaces (su tráfico se suma a la ruta alterna
    de menor tráfico) → bloqueo de equipos (si es un servidor, el otro absorbe su
    carga: +20 % de CPU y +15 Mbps en sus enlaces) → recorrido BFS con el AFD →
    aislamiento de los bloqueados → actualización del peso de todos los enlaces.
    """
    g = copy.deepcopy(g_cruda)
    semilla = g.graph["semilla"]
    habilitar = set(habilitar)
    g.graph["auto_bloqueo"] = set(fallas)
    for v in g.nodes:
        g.nodes[v].update(bloqueo_manual=False, bloqueo_activo=False, origen_bloqueo=None, estado_afd=au.NORMAL)
    for u, v in g.edges:
        g.edges[u, v].update(estado_enlace="Activa", motivo_enlace=None)
    for nodo in fallas:
        inyectar_falla(g, nodo, _rng("falla_manual", semilla, nodo))

    cortes = [(e, "Aleatoria al iniciar") for e in interrupcion_inicial] + [(e, "Manual") for e in enlaces_bloqueados]
    for (u, v), motivo in cortes:
        if not g.has_edge(u, v) or not g.edges[u, v]["activa"]:
            continue
        trafico = g.edges[u, v]["trafico"]
        g.edges[u, v].update(activa=False, estado_enlace="Interrumpida", motivo_enlace=motivo)
        try:
            ruta = nx.shortest_path(subgrafo_activo(g), u, v, weight="trafico", method="dijkstra")
            for a, b in zip(ruta, ruta[1:]):
                e = g.edges[a, b]
                _fijar_trafico(e, e["trafico"] + trafico)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    todos = [(b, "Aleatorio al iniciar") for b in bloqueo_inicial] + \
            [(b, "Manual (BLQ)") for b in nodos_bloqueados if b not in bloqueo_inicial]
    for b, origen in todos:
        d = g.nodes[b]
        d.update(bloqueo_manual=True, origen_bloqueo=origen)      # su cadena incluye BLQ
        if b in habilitar:                                         # BLQ … HAB → vuelve a Normal
            continue
        d["bloqueo_activo"] = True
        _desactivar(g, b)
        if d["tipo"] == "Servidor":
            for s in SERVIDORES:
                if s != b and s not in {x for x, _ in todos if x not in habilitar}:
                    g.nodes[s]["cpu"] = float(min(100.0, g.nodes[s]["cpu"] + 20))
                    for v in g.neighbors(s):
                        e = g.edges[s, v]
                        if e["activa"]:
                            _fijar_trafico(e, e["trafico"] + 15.0)

    orden = recorrer_red(g, habilitar)
    for v in ORDEN:
        d = g.nodes[v]
        d["motivo_bloqueo"] = None
        if d["estado_afd"] == au.BLOQUEADO:
            d["motivo_bloqueo"] = (d["origen_bloqueo"] if d["bloqueo_activo"] else
                                   "Sin servicio" if d["sin_servicio"] else "AFD: 2 fallas seguidas")
            _desactivar(g, v)
    for u, v, e in g.edges(data=True):
        if e["estado_enlace"] == "Activa" and not e["activa"]:
            e.update(estado_enlace="Inactiva", motivo_enlace="Equipo bloqueado")
    for v in ORDEN:
        g.nodes[v]["estado"] = g.nodes[v]["estado_afd"]
    dt.actualizar_pesos(g)          # el peso refleja el tráfico final y los estados del AFD
    g.graph["orden_recorrido"] = orden
    g.graph["escenario"] = {"nodos": list(nodos_bloqueados), "enlaces": [list(e) for e in enlaces_bloqueados],
                            "fallas": list(fallas), "habilitar": sorted(habilitar),
                            "bloqueo_inicial": list(bloqueo_inicial),
                            "interrupcion_inicial": [list(e) for e in interrupcion_inicial]}
    return g


def red_de_sesion(semilla: int, proporcion_fallas: float = 0.12) -> tuple[nx.Graph, dict, nx.Graph]:
    """Red cruda, configuración aleatoria y red evaluada que ve el usuario al iniciar la sesión."""
    cruda = generar_red(semilla, proporcion_fallas)
    conf = configuracion_inicial(semilla)
    base = simular(cruda, bloqueo_inicial=conf["bloqueados"], interrupcion_inicial=conf["interrumpidos"])
    return cruda, conf, base


def nodos_activos(g: nx.Graph) -> list[str]:
    return [v for v in ORDEN if g.nodes[v].get("estado_afd") != au.BLOQUEADO]


def subgrafo_activo(g: nx.Graph) -> nx.Graph:
    h = g.subgraph(nodos_activos(g)).copy()
    h.remove_edges_from([(u, v) for u, v, a in h.edges(data=True) if not a["activa"]])
    return h


# --------------------------------------------------------------------------
# 5. Dijkstra (camino de menor costo) y rutas de referencia
# --------------------------------------------------------------------------
PESO = "peso"


def grafo_costos(g: nx.Graph) -> nx.DiGraph:
    """Grafo dirigido de los enlaces utilizables; cada sentido lleva el peso de la arista."""
    h = subgrafo_activo(g)
    dg = nx.DiGraph()
    dg.add_nodes_from(h.nodes)
    for u, v, a in h.edges(data=True):
        dg.add_edge(u, v, costo=a[PESO])
        dg.add_edge(v, u, costo=a[PESO])
    return dg


def costo_ruta(g: nx.Graph, ruta: list[str] | None) -> float:
    """Costo acumulado de una ruta: suma del atributo 'peso' de sus aristas."""
    return float(sum(g.edges[a, b][PESO] for a, b in zip(ruta, ruta[1:]))) if ruta else float("nan")


def validar_extremos(g: nx.Graph, origen: str, destino: str) -> str | None:
    """Mensaje de error si el par origen–destino no es válido; None si se puede buscar ruta."""
    if origen == destino:
        return "El origen y el destino deben ser dispositivos diferentes."
    for rol, v in (("origen", origen), ("destino", destino)):
        if v not in g:
            return f"El {rol} {v} no existe en la red."
        if g.nodes[v].get("estado_afd") == au.BLOQUEADO:
            motivo = g.nodes[v].get("motivo_bloqueo") or "Bloqueado"
            return f"El {rol} {v} está Bloqueado ({motivo}) y no puede participar en el recorrido."
    return None


def dijkstra(g: nx.Graph, origen: str, destino: str) -> dict:
    """ALGORITMO DE DIJKSTRA con NetworkX sobre los enlaces activos y los equipos no bloqueados.

    Usa nx.shortest_path(..., weight="peso", method="dijkstra"): sin el parámetro
    weight el resultado sería la ruta de menos saltos.
    """
    error = validar_extremos(g, origen, destino)
    if error:
        return {"ruta": None, "error": error}
    h = subgrafo_activo(g)
    try:
        ruta = nx.shortest_path(h, origen, destino, weight=PESO, method="dijkstra")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return {"ruta": None, "error": f"No existe un camino disponible entre {origen} y {destino} "
                                       "con los enlaces activos."}
    return resumen_ruta(g, ruta) | {"error": None}


def resumen_ruta(g: nx.Graph, ruta: list[str] | None) -> dict:
    """Latencia total (ms), tráfico total (Mbps), costo (peso), saltos y estado de los equipos de una ruta."""
    if not ruta:
        return {"ruta": None, "costo": None, "trafico": None, "latencia": None, "saltos": None,
                "estados": None, "anomalos": None}
    pares = list(zip(ruta, ruta[1:]))
    return {"ruta": list(ruta), "costo": costo_ruta(g, ruta),
            "estados": [g.nodes[v]["estado"] for v in ruta],
            "anomalos": [v for v in ruta if g.nodes[v]["estado"] == au.ANOMALO],
            "trafico": float(sum(g.edges[a, b]["trafico"] for a, b in pares)),
            "latencia": float(sum(g.edges[a, b]["latencia"] for a, b in pares)),
            "saltos": len(pares)}


def ruta_optima(g: nx.Graph, origen: str, destino: str) -> list[str] | None:
    """Camino de menor costo (Dijkstra sobre 'peso'): referencia que la GNN aprende a imitar."""
    return dijkstra(g, origen, destino)["ruta"]


def ruta_saltos(g: nx.Graph, origen: str, destino: str) -> list[str] | None:
    """Línea base: el camino con menos saltos (BFS sin weight), que ignora el peso."""
    if validar_extremos(g, origen, destino):
        return None
    try:
        return nx.shortest_path(subgrafo_activo(g), origen, destino)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def metricas_red(g: nx.Graph) -> dict:
    h = subgrafo_activo(g)
    clientes = [v for v in ORDEN if g.nodes[v]["tipo"] != "Servidor"]
    con_servicio = [v for v in clientes if v in h and not g.nodes[v]["sin_servicio"]]
    traf = [a["trafico"] for _, _, a in h.edges(data=True)]
    estados = [g.nodes[v]["estado"] for v in g]
    return {
        **{e: estados.count(e) for e in au.ESTADOS},
        "enlaces_activos": h.number_of_edges(),
        "componentes": nx.number_connected_components(h) if len(h) else 0,
        "servicio_pct": 100.0 * len(con_servicio) / len(clientes),
        "trafico_medio": float(np.mean(traf)) if traf else 0.0,
        "congestionados": int(sum(t >= dt.TRAFICO_ALTO for t in traf)),
        "trafico_total": float(sum(traf)),
    }
