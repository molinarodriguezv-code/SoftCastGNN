"""Pruebas de la red fija, el recorrido BFS, la simulación y la GNN de rutas."""

import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import automata as au  # noqa: E402
from core import gnn  # noqa: E402
from core import red as rd  # noqa: E402

MODELO = gnn.cargar(Path(__file__).resolve().parents[1] / "modelos" / "gnn_rutas").modelo


def test_inventario_fijo():
    g = rd.construir_topologia(5)
    tipos = [g.nodes[v]["tipo"] for v in g]
    assert [tipos.count(t) for t in ("Servidor", "PC", "Portátil", "Impresora", "IoT")] == [2, 5, 5, 2, 4]
    assert len(g) == 18 and g.number_of_edges() == 29 and nx.is_connected(g)


def test_sin_enlaces_duplicados_ni_lazos():
    pares = [frozenset((u, v)) for u, v, _ in rd.ENLACES_FIJOS]
    assert len(pares) == len(set(pares)) == 29
    assert all(len(p) == 2 for p in pares)


def test_cada_equipo_tiene_tres_o_cuatro_enlaces():
    g = rd.construir_topologia(5)
    assert all(3 <= g.degree(v) <= 4 for v in g)
    assert {v for v in g if g.degree(v) == 4} == {"S1", "S2", "PT1", "PT5"}
    assert nx.is_biconnected(g)                       # ningún equipo es punto único de falla
    assert nx.node_connectivity(g) == 3 and nx.edge_connectivity(g) == 3


def test_bloquear_cualquier_equipo_o_enlace_no_desconecta_la_red():
    g = rd.construir_topologia(5)
    for v in g:
        assert nx.is_connected(g.subgraph(set(g) - {v}))
    for e in g.edges:
        h = g.copy()
        h.remove_edge(*e)
        assert nx.is_connected(h)


def test_aristas_con_trafico_latencia_y_peso():
    from core import datos as dt
    g = rd.construir_topologia(5)
    for _, _, a in g.edges(data=True):
        lo, hi = dt.ENLACES[a["clase"]]["trafico"]
        assert lo <= a["trafico"] <= hi
        assert set(a) == {"clase", "activa", "latencia", "trafico", "peso"}
        assert a["peso"] > 0
        assert abs(a["peso"] - (0.30 * a["trafico"] / 140 + 0.70 * a["latencia"] / 15)) < 1e-3   # 30 % / 70 %


def test_peso_penaliza_extremos_anomalos_y_se_actualiza():
    from core import datos as dt
    s = rd.simular(rd.generar_red(3))
    for u, v, a in s.edges(data=True):
        esperado = dt.calcular_peso(a["trafico"], a["latencia"], dt.extremos_anomalos(s, u, v))
        assert abs(a["peso"] - esperado) < 1e-3
    base = rd.simular(rd.generar_red(5, 0.0))
    con_falla = rd.simular(rd.generar_red(5, 0.0), fallas=["PC4"])
    assert all(con_falla.edges["PC4", v]["peso"] > base.edges["PC4", v]["peso"] for v in base.neighbors("PC4"))


def test_dijkstra_usa_el_peso_y_no_los_saltos():
    import itertools
    g = rd.simular(rd.generar_red(3))
    h = rd.subgrafo_activo(g)
    distintos = 0
    for o, t in itertools.permutations(rd.nodos_activos(g), 2):
        r = rd.dijkstra(g, o, t)
        assert r["error"] is None
        assert abs(r["costo"] - nx.dijkstra_path_length(h, o, t, weight="peso")) < 1e-9
        assert r["saltos"] >= nx.shortest_path_length(h, o, t)
        distintos += r["costo"] < rd.costo_ruta(g, rd.ruta_saltos(g, o, t)) - 1e-9
    assert distintos > 0      # en varios pares la ruta de menor costo no es la de menos saltos


def test_dijkstra_valida_origen_destino():
    g = rd.simular(rd.generar_red(5, 0.0), ["PC1"])
    assert rd.dijkstra(g, "PT1", "PT1")["error"].startswith("El origen y el destino")
    assert "Bloqueado" in rd.dijkstra(g, "PC1", "S2")["error"]
    aislado = rd.simular(rd.generar_red(5, 0.0), [], [("PT1", "PC1"), ("PT1", "IM1"), ("PT1", "PC2"), ("PT1", "IoT1")])
    assert rd.dijkstra(aislado, "PT1", "S2")["ruta"] is None
    assert gnn.buscar_ruta(MODELO, g, "PC1", "S2")["ruta"] is None


def test_misma_topologia_distinto_trafico():
    a, b = rd.generar_red(1), rd.generar_red(2)
    assert set(a.edges) == set(b.edges)
    assert [a.edges[e]["trafico"] for e in a.edges] != [b.edges[e]["trafico"] for e in a.edges]


def test_recorrido_bfs():
    g = rd.simular(rd.generar_red(5, 0.0))
    visitados = [o for o in g.graph["orden_recorrido"] if o["paso"] is not None]
    assert [o["nodo"] for o in visitados[:2]] == ["S1", "S2"]
    assert [o["nivel"] for o in visitados] == sorted(o["nivel"] for o in visitados)
    assert g.nodes["IM1"]["padre"] == "S1" and g.nodes["IM1"]["nivel"] == 1
    assert max(o["nivel"] for o in visitados) <= 3


def test_bloqueo_de_servidor():
    cruda = rd.generar_red(5, 0.0)
    s = rd.simular(cruda, ["S1"])
    assert s.nodes["S1"]["estado"] == au.BLOQUEADO
    assert not s.nodes["PC1"]["sin_servicio"]          # sigue llegando a S2 por PC4
    assert rd.ruta_optima(s, "PT1", "S1") is None
    assert "bloqueo_manual" not in cruda.nodes["S1"]   # la red original no cambia


def test_ruta_optima_esquiva_bloqueados():
    s = rd.simular(rd.generar_red(5, 0.0), ["PC1"])
    ruta = rd.ruta_optima(s, "PT1", "S2")
    assert ruta[0] == "PT1" and ruta[-1] == "S2" and "PC1" not in ruta


def test_gradiente_analitico_coincide_con_numerico():
    esc = gnn.preparar(rd.simular(rd.generar_red(3)))
    m = gnn.GNNRutas(semilla=1)
    _, grads = m.perdida_y_gradientes(esc)
    eps = 1e-6
    for clave in ("W1", "w2", "b1"):
        p = m.params[clave]
        idx = tuple(np.unravel_index(np.argmax(np.abs(grads[clave])), p.shape))
        orig = p[idx]
        p[idx] = orig + eps
        mas, _ = m.perdida_y_gradientes(esc)
        p[idx] = orig - eps
        menos, _ = m.perdida_y_gradientes(esc)
        p[idx] = orig
        assert abs((mas - menos) / (2 * eps) - grads[clave][idx]) < 1e-5


def test_gnn_entrenada_encuentra_rutas_optimas():
    escs = [gnn.preparar(rd.simular(rd.generar_red(s))) for s in range(900, 910)]
    m = gnn.evaluar(MODELO, escs)
    assert m["optimas"] > 0.95 and m["sobrecosto_medio"] < 0.01
    assert m["optimas"] > gnn.evaluar(None, escs)["optimas"]


def test_buscar_ruta_y_rendimiento():
    g = rd.simular(rd.generar_red(5))
    inicio = time.perf_counter()
    r = gnn.buscar_ruta(MODELO, g, "PT1", "S2")
    assert time.perf_counter() - inicio < 2.0                       # RNF-02
    assert r["ruta"][0] == "PT1" and r["ruta"][-1] == "S2"
    assert all(g.has_edge(a, b) for a, b in zip(r["ruta"], r["ruta"][1:]))


# ------------------------------------------------------------------ ponderación 30 % / 70 %
def test_ponderacion_prioriza_la_latencia():
    from core import datos as dt
    assert (dt.ALFA_TRAFICO, dt.BETA_LATENCIA) == (0.30, 0.70)
    # un milisegundo pesa más que un Mbps normalizado: subir la latencia de 2 a 12 ms cuesta más que
    # subir el tráfico de 10 a 110 Mbps
    assert dt.calcular_peso(10, 12) > dt.calcular_peso(110, 2)
    # normalización: los máximos de los rangos de generación valen 1
    assert dt.normalizados(140, 15) == (1.0, 1.0)


# ------------------------------------------------------------------ configuración aleatoria de la sesión
def test_configuracion_inicial_dos_bloqueados_distintos():
    for s in range(200):
        conf = rd.configuracion_inicial(s)
        assert len(conf["bloqueados"]) == 2 and len(set(conf["bloqueados"])) == 2
        assert set(conf["bloqueados"]) != {"S1", "S2"}
        assert 1 <= len(conf["interrumpidos"]) <= 3
        assert all(u not in conf["bloqueados"] and v not in conf["bloqueados"] for u, v in conf["interrumpidos"])
    assert rd.configuracion_inicial(7) == rd.configuracion_inicial(7)            # reproducible
    assert len({tuple(rd.configuracion_inicial(s)["bloqueados"]) for s in range(30)}) > 10   # varía


def test_red_de_sesion_exactamente_dos_bloqueados_y_red_activa_conectada():
    for s in range(60):
        _, conf, base = rd.red_de_sesion(s, 0.2)
        bloqueados = {v for v in base if base.nodes[v]["estado"] == au.BLOQUEADO}
        assert bloqueados == set(conf["bloqueados"])
        assert {base.nodes[v]["estado"] for v in base} <= {au.NORMAL, au.ANOMALO, au.BLOQUEADO}
        assert nx.is_connected(rd.subgrafo_activo(base))                          # integridad
        for u, v, e in base.edges(data=True):
            if (u, v) in conf["interrumpidos"]:
                assert e["estado_enlace"] == "Interrumpida" and not e["activa"]
            elif u in bloqueados or v in bloqueados:
                assert e["estado_enlace"] == "Inactiva" and not e["activa"]
            else:
                assert e["estado_enlace"] == "Activa" and e["activa"]


def test_habilitar_un_equipo_bloqueado_al_inicio():
    cruda, conf, _ = rd.red_de_sesion(11, 0.2)
    b = conf["bloqueados"][0]
    s = rd.simular(cruda, habilitar=[b], bloqueo_inicial=conf["bloqueados"], interrupcion_inicial=conf["interrumpidos"])
    assert s.nodes[b]["estado"] == au.NORMAL and s.nodes[b]["cadena"][-1] == au.HAB


def test_rutas_excluyen_bloqueados_e_inactivos_y_dan_latencia_en_ms():
    import itertools
    for semilla in (1, 2, 3):
        _, _, g = rd.red_de_sesion(semilla, 0.2)
        for o, t in itertools.permutations(rd.nodos_activos(g), 2):
            r = rd.dijkstra(g, o, t)
            assert r["error"] is None
            assert all(g.nodes[v]["estado"] != au.BLOQUEADO for v in r["ruta"])
            assert all(g.edges[a, b]["activa"] for a, b in zip(r["ruta"], r["ruta"][1:]))
            lat = sum(g.edges[a, b]["latencia"] for a, b in zip(r["ruta"], r["ruta"][1:]))
            assert abs(r["latencia"] - lat) < 1e-9 and r["latencia"] != r["costo"]
            assert gnn.buscar_ruta(MODELO, g, o, t)["ruta"] is not None


def test_equipo_anomalo_transitable_si_es_la_unica_alternativa():
    from core import datos as dt
    g = rd.simular(rd.generar_red(5, 0.0))
    g.nodes["PT5"]["estado"] = g.nodes["PT5"]["estado_afd"] = au.ANOMALO
    for v in ("PC3", "PT2"):                                   # IoT2 solo puede salir por PT5
        g.edges["IoT2", v]["activa"] = False
    dt.actualizar_pesos(g)
    r = rd.dijkstra(g, "IoT2", "S2")
    assert r["error"] is None and "PT5" in r["ruta"] and r["anomalos"] == ["PT5"]
    assert gnn.buscar_ruta(MODELO, g, "IoT2", "S2")["ruta"][1] == "PT5"


def test_sin_ruta_disponible_se_informa():
    g = rd.simular(rd.generar_red(5, 0.0))
    for v in list(g.neighbors("IoT2")):
        g.edges["IoT2", v]["activa"] = False
    r = rd.dijkstra(g, "IoT2", "S2")
    assert r["ruta"] is None and "No existe un camino disponible" in r["error"]
