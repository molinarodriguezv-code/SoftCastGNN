"""SoftCast · Red de computadores: estados con AFD y camino óptimo con una GNN.

Prototipo funcional (Streamlit):
    * Red fija: los 18 equipos son los nodos y los 29 enlaces las aristas; cada equipo tiene 3 o 4 enlaces.
    * Cada arista guarda tráfico (Mbps), latencia (ms), disponibilidad y su 'peso' para Dijkstra:
      peso = 0,30·T/140 + 0,70·L/15 + 0,2 por extremo Anómalo (la latencia tiene prioridad).
    * Cada sesión nueva (o al refrescar la página) genera una configuración aleatoria: dos equipos
      Bloqueados y de uno a tres enlaces interrumpidos; se conserva en st.session_state.
    * Dijkstra (NetworkX, weight="peso") calcula el camino de menor costo; una animación muestra el recorrido.
    * Recorrido en anchura (BFS) desde los servidores.
    * AFD de tres estados (Normal, Anómalo, Bloqueado) para cada dispositivo.
    * GNN tipo Bellman-Ford neuronal que encuentra el camino óptimo entre dos equipos.
    * Simulación de bloqueo y habilitación de nodos y enlaces e inyección de fallas.
    * Botón «Iniciar recorrido» (o tecla Enter) en el panel lateral.

Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import secrets

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core import automata as au
from core import datos as dt
from core import gnn
from core import red as rd
from core import visual

st.set_page_config(page_title="SoftCast · Camino óptimo", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
.hero { background: linear-gradient(120deg, #0F2A3D 0%, #111A2B 55%, #1A1530 100%);
        border: 1px solid #1F2B40; border-radius: 16px; padding: 22px 26px; margin-bottom: 18px; }
.hero h1 { font-size: 1.7rem; margin: 0 0 4px 0; color: #F1F5F9; font-weight: 700; }
.hero p { color: #94A3B8; margin: 0; font-size: 0.95rem; }
.hero .tag { display:inline-block; background:#0E3B3A; color:#5EEAD4; border:1px solid #115E59;
             padding:2px 10px; border-radius:999px; font-size:0.75rem; margin-right:6px; }
.kpi { background:#111A2B; border:1px solid #1F2B40; border-radius:14px; padding:14px 16px; }
.kpi .lbl { color:#94A3B8; font-size:0.76rem; text-transform:uppercase; letter-spacing:.06em; }
.kpi .val { color:#F1F5F9; font-size:1.6rem; font-weight:700; line-height:1.2; }
.kpi .sub { color:#64748B; font-size:0.78rem; }
.kpi.ok .val { color:#4ADE80; } .kpi.alerta .val { color:#F87171; } .kpi.gris .val { color:#94A3B8; }
.kpi.info .val { color:#5EEAD4; } .kpi.ambar .val { color:#FBBF24; }
.leyenda { display:flex; flex-wrap:wrap; gap:14px; font-size:0.85rem; color:#CBD5E1; margin:6px 0 10px; }
.res { background:#0F172A; border:1px solid #1E3A5F; border-radius:14px; padding:14px 18px; margin:6px 0 12px 0; }
.res-h { color:#94A3B8; font-size:.9rem; margin-bottom:8px; }
.res-ruta { margin-bottom:10px; line-height:2.1; }
.chipr { border:2px solid; border-radius:999px; padding:3px 10px; margin:0 1px; white-space:nowrap; }
.res-k { display:flex; flex-wrap:wrap; gap:28px; }
.res-k .lbl { color:#94A3B8; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
.res-k .big { color:#22D3EE; font-size:2.1rem; font-weight:800; }
.res-k .mid { color:#E6EDF3; font-size:1.35rem; font-weight:700; margin-top:6px; }
.res-k .sub { color:#64748B; font-size:.72rem; }
.res .nota { margin-top:10px; color:#FCA5A5; font-size:.9rem; }
.leyenda span.c { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.leyenda span.l { display:inline-block; width:22px; height:4px; border-radius:2px; margin-right:6px; vertical-align:3px; }
.chip { display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:0.9rem; }
.chip.Normal { background:#052E1A; color:#4ADE80; border:1px solid #14532D; }
.chip.Anómalo { background:#3B0A0A; color:#F87171; border:1px solid #7F1D1D; }
.chip.Bloqueado { background:#1E293B; color:#CBD5E1; border:1px solid #475569; }
.ruta { font-family:'JetBrains Mono', monospace; font-size:1.0rem; color:#22D3EE; }
.mono { font-family:'JetBrains Mono', monospace; font-size:0.86rem; }
.card { background:#111A2B; border:1px solid #1F2B40; border-radius:14px; padding:16px 18px; }
div[data-testid="stTabs"] button p { font-size:0.95rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

CARPETA_MODELOS = Path(__file__).parent / "modelos"
COLOR_ESTADO = {"Normal": "color:#4ADE80", "Anómalo": "color:#F87171;font-weight:600",
                "Bloqueado": "color:#94A3B8;font-weight:600"}
NODOS = rd.ORDEN


@st.cache_resource(show_spinner="Cargando la GNN entrenada…")
def obtener_modelo() -> gnn.ResultadoEntrenamiento:
    try:
        return gnn.cargar(CARPETA_MODELOS / "gnn_rutas")
    except (FileNotFoundError, KeyError, ValueError):
        return gnn.entrenar(gnn.generar_dataset(300, 2026))


@st.cache_data
def red_cruda(semilla: int, fallas: float):
    return rd.generar_red(semilla, fallas)


def kpi(etiqueta, valor, sub="", clase=""):
    return (f'<div class="kpi {clase}"><div class="lbl">{etiqueta}</div><div class="val">{valor}</div>'
            f'<div class="sub">{sub}</div></div>')


def leyenda(recorrido: bool = False) -> None:
    c = visual.COLORES
    nodos = (f'<span><span class="c" style="background:{c["Normal"]}"></span>Normal</span>'
             f'<span><span class="c" style="background:{c["Anómalo"]}"></span>Anómalo</span>'
             f'<span><span class="c" style="background:{c["Bloqueado"]}"></span>Bloqueado</span>')
    if recorrido == "animacion":
        extra = (f'<span><span class="l" style="background:{c["cambio"]}"></span>Nodo y enlace que se recorren</span>'
                 f'<span><span class="l" style="background:{c["ruta"]}"></span>Tramo ya recorrido</span>'
                 f'<span><span class="l" style="background:{c["traf_bajo"]}"></span>Aún no recorrido (color por tráfico)</span>'
                 f'<span><span class="c" style="background:transparent;border:3px solid {c["falla"]}"></span>Falla inyectada</span>')
    elif recorrido:
        extra = (f'<span><span class="c" style="background:{c["pendiente"]};border:1px solid #475569"></span>Aún no visitado</span>'
                 f'<span><span class="c" style="background:transparent;border:3px solid {c["actual"]}"></span>Nodo actual</span>')
    else:
        extra = (f'<span><span class="l" style="background:{c["traf_bajo"]}"></span>Tráfico &lt; 30 Mbps</span>'
                 f'<span><span class="l" style="background:{c["traf_medio"]}"></span>30–60 Mbps</span>'
                 f'<span><span class="l" style="background:{c["traf_alto"]}"></span>≥ 60 Mbps</span>'
                 f'<span><span class="l" style="background:{c["ruta"]}"></span>Ruta seleccionada</span>'
                 f'<span><span class="c" style="background:transparent;border:3px solid {c["falla"]}"></span>Falla inyectada</span>'
                 f'<span><span class="c" style="background:transparent;border:3px solid {c["cambio"]}"></span>Cambió de estado</span>')
    enlaces_ = (f'<span><span class="l" style="background:transparent;border-top:3px dashed {c["interrumpida"]};height:0">'
                f'</span>Enlace interrumpido ✕</span>'
                f'<span><span class="l" style="background:transparent;border-top:3px dotted {c["inactiva"]};height:0">'
                f'</span>Inactivo (equipo bloqueado 🔒)</span>')
    st.markdown('<div class="leyenda">' + nodos + extra + enlaces_ +
                '<span>◆ Servidor · ● PC / Portátil · ■ Impresora · ▲ IoT · color y grosor de la arista = tráfico; '
                'etiqueta = latencia (ms) y/o tráfico (Mbps)</span></div>', unsafe_allow_html=True)


def tema(fig: go.Figure, alto: int = 330) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=alto, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=30),
                      font=dict(family="Inter, sans-serif", color="#CBD5E1"),
                      legend=dict(orientation="h", y=-0.3, x=0))
    fig.update_xaxes(gridcolor="#1F2B40")
    fig.update_yaxes(gridcolor="#1F2B40")
    return fig


def tabla_enlaces(g) -> pd.DataFrame:
    return pd.DataFrame([{"Enlace": f"{u} — {v}", "Tipo": a["clase"], "Latencia (ms)": a["latencia"],
                          "Tráfico (Mbps)": a["trafico"], "Peso": a["peso"],
                          "Estado": a.get("estado_enlace", "Activa")}
                         for u, v, a in g.edges(data=True)])


def comparar_rutas(g, origen, destino, modelo):
    """Dijkstra (referencia), GNN y ruta de menos saltos: latencia, tráfico, peso, saltos y diferencias."""
    r = gnn.buscar_ruta(modelo, g, origen, destino)
    dij = rd.dijkstra(g, origen, destino)
    filas = []
    for nombre, ruta in (("Dijkstra (óptimo)", dij["ruta"]), ("GNN (red neuronal)", r["ruta"]),
                         ("Menos saltos (BFS)", rd.ruta_saltos(g, origen, destino))):
        x = rd.resumen_ruta(g, ruta)
        dif = (None if not ruta or not dij["ruta"] else 100 * (x["costo"] - dij["costo"]) / dij["costo"])
        filas.append({"Método": nombre, "Ruta": " → ".join(ruta) if ruta else "—", "Saltos": x["saltos"],
                      "Latencia total (ms)": x["latencia"], "Tráfico total (Mbps)": x["trafico"],
                      "Peso (costo)": x["costo"], "Anómalos en la ruta": ", ".join(x["anomalos"]) if ruta and x["anomalos"] else ("—" if not ruta else "ninguno"),
                      "Diferencia de peso vs. Dijkstra (%)": dif,
                      "¿Óptima?": "—" if dif is None else ("Sí" if abs(dif) < 1e-6 else "No")})
    return r, dij, pd.DataFrame(filas)


def mostrar_comparacion(comp: pd.DataFrame) -> None:
    st.dataframe(comp.style.format({"Peso (costo)": "{:.3f}", "Tráfico total (Mbps)": "{:.1f}",
                                    "Latencia total (ms)": "{:.2f}", "Saltos": "{:.0f}",
                                    "Diferencia de peso vs. Dijkstra (%)": "{:+.2f}"}, na_rep="—")
                 .map(lambda x: "color:#4ADE80" if x == "Sí" else ("color:#F87171" if x == "No" else ""),
                      subset=["¿Óptima?"]), hide_index=True, width="stretch")


def tarjeta_ruta(g, res: dict, origen: str, destino: str, metodo: str = "Dijkstra") -> None:
    """Cuadro de resultados: origen, destino, ruta en orden con el estado de cada equipo y latencia total en ms."""
    if not res.get("ruta"):
        st.error(f"**No existe una ruta disponible** de {origen} a {destino}. {res.get('error') or res.get('motivo') or ''}")
        return
    col = {"Normal": "#22C55E", "Anómalo": "#EF4444", "Bloqueado": "#64748B"}
    chips = " <span style='color:#475569'>→</span> ".join(
        f"<span class='chipr' style='border-color:{col[e]}'><b>{v}</b> <small style='color:{col[e]}'>{e}</small></span>"
        for v, e in zip(res["ruta"], res["estados"]))
    nota = ""
    medio = [v for v in res["ruta"][1:-1] if v in res["anomalos"]]
    extremos = [v for v in (res["ruta"][0], res["ruta"][-1]) if v in res["anomalos"]]
    if medio:
        nota += (f"<div class='nota'>⚠ La ruta atraviesa {', '.join(medio)} (Anómalo): sigue siendo transitable y su "
                 f"penalización (+0,2 en cada uno de sus enlaces) ya está incluida en el peso; aun así es la mejor "
                 f"alternativa disponible.</div>")
    if extremos:
        nota += (f"<div class='nota'>⚠ {' y '.join(extremos)} (origen o destino) está Anómalo: la comunicación es posible, "
                 f"pero el equipo está en observación.</div>")
    st.markdown(f"""
<div class="res">
  <div class="res-h">Resultado · {metodo} &nbsp;|&nbsp; Origen <b>{origen}</b> → Destino <b>{destino}</b></div>
  <div class="res-ruta">{chips}</div>
  <div class="res-k">
    <div><div class="lbl">Latencia total acumulada</div><div class="big">{res['latencia']:.2f} ms</div></div>
    <div><div class="lbl">Tráfico total acumulado</div><div class="mid">{res['trafico']:.1f} Mbps</div></div>
    <div><div class="lbl">Peso (costo interno de Dijkstra)</div><div class="mid">{res['costo']:.3f}</div>
         <div class="sub">adimensional · no son ms</div></div>
    <div><div class="lbl">Saltos</div><div class="mid">{res['saltos']}</div></div>
  </div>{nota}
</div>""", unsafe_allow_html=True)


def explicar_diferencia(dij: dict, r: dict, g) -> None:
    if not dij["ruta"] or not r["ruta"]:
        return
    if r["ruta"] == dij["ruta"]:
        st.success(f"La GNN encontró exactamente la misma ruta que Dijkstra (costo {dij['costo']:.3f}).")
        return
    c = rd.costo_ruta(g, r["ruta"])
    if abs(c - dij["costo"]) < 1e-9:
        st.info("La GNN eligió una ruta distinta con el mismo costo que Dijkstra (empate).")
    else:
        st.warning(f"La GNN eligió {' → '.join(r['ruta'])} con costo {c:.3f}, "
                   f"{100 * (c - dij['costo']) / dij['costo']:+.2f} % sobre el óptimo de Dijkstra ({dij['costo']:.3f}).")


# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ SoftCast NOC")
    st.caption("Lenguajes y Autómatas · UdeC Fusagasugá · 802M")
    st.markdown("### Red fija · 18 equipos")
    st.markdown("2 servidores · 5 PC · 5 portátiles · 2 impresoras · 4 IoT · 29 enlaces (3 o 4 por equipo)")
    if "semilla" not in st.session_state:               # nueva sesión o página refrescada → configuración nueva
        st.session_state["semilla"] = secrets.randbelow(10000)

    def nueva_configuracion():
        st.session_state["semilla"] = secrets.randbelow(10000)
        st.session_state["recorrido"] = st.session_state["error_recorrido"] = None

    semilla = st.number_input("Semilla de la configuración", 0, 9999, step=1, key="semilla",
                              help="Se sortea al iniciar la sesión o al refrescar la página. Define el tráfico, las "
                                   "latencias, la CPU, los eventos, los dos equipos bloqueados y los enlaces "
                                   "interrumpidos. Escriba una semilla para reproducir una configuración.")
    st.button("🎲 Nueva configuración aleatoria", on_click=nueva_configuracion, width="stretch",
              help="Equivale a refrescar la página: sortea otra semilla.")
    prop_fallas = st.slider("Proporción de fallas iniciales", 0.0, 0.3, 0.20, step=0.02)
    contenido = {"Latencia (ms)": "latencia", "Tráfico (Mbps)": "trafico", "Ambos": "ambos"}[
        st.radio("Etiqueta de los enlaces", ["Latencia (ms)", "Tráfico (Mbps)", "Ambos"], index=2, horizontal=True,
                 key="etiqueta_enlaces")]
    st.markdown("### Recorrido")
    origen = st.selectbox("Origen", NODOS, index=NODOS.index("IM1"), key="origen")
    destino = st.selectbox("Destino", NODOS, index=NODOS.index("PC5"), key="destino")
    iniciar = st.button("▶ Iniciar recorrido", type="primary", shortcut="Enter", width="stretch",
                        help="Calcula el camino óptimo con Dijkstra y lo anima en la pestaña Simulación (tecla Enter).")
    aviso = st.empty()
    st.caption("Peso = 0,30·tráfico/140 Mbps + 0,70·latencia/15 ms (+0,2 por extremo Anómalo) · "
               "Dijkstra (NetworkX, weight=\"peso\") · GNN Bellman-Ford neuronal de 32 neuronas.")
    st.divider()
    st.caption("Paula Andrea Merchán Suaza · Victor Hernando Molina Rodríguez")

resultado = obtener_modelo()
modelo = resultado.modelo
cruda = red_cruda(int(semilla), float(prop_fallas))
conf = rd.configuracion_inicial(int(semilla))           # dos equipos bloqueados y enlaces interrumpidos al azar
base = rd.simular(cruda, bloqueo_inicial=conf["bloqueados"], interrupcion_inicial=conf["interrumpidos"])
m_base = rd.metricas_red(base)
clave = (int(semilla), float(prop_fallas))
if st.session_state.get("clave_red") != clave:
    st.session_state["clave_red"] = clave
    for k in ("sim_nodos", "sim_enlaces", "sim_fallas", "sim_hab"):
        st.session_state[k] = []

# Escenario de simulación (sus controles están en la pestaña Simulación y se conservan en la sesión)
ENLACES_RED = {f"{u} — {v}": (u, v) for u, v in base.edges}
bloq_afd = [v for v in NODOS if base.nodes[v]["motivo_bloqueo"]]
esc = rd.simular(cruda, st.session_state.get("sim_nodos", []),
                 [ENLACES_RED[e] for e in st.session_state.get("sim_enlaces", []) if e in ENLACES_RED],
                 st.session_state.get("sim_fallas", []),
                 [v for v in st.session_state.get("sim_hab", []) if v in bloq_afd],
                 conf["bloqueados"], conf["interrumpidos"])
m_esc = rd.metricas_red(esc)
cambios = {v for v in NODOS if esc.nodes[v]["estado"] != base.nodes[v]["estado"]}

if iniciar:                                         # botón o tecla Enter
    validacion = rd.dijkstra(esc, origen, destino)
    if validacion["error"]:
        st.session_state["recorrido"] = None            # no se ejecuta una animación inválida
        st.session_state["error_recorrido"] = (origen, destino, validacion["error"])
        aviso.error(validacion["error"])
    else:
        st.session_state["error_recorrido"] = None
        st.session_state["recorrido"] = (origen, destino)
        st.session_state["autoplay"] = True
rec = st.session_state.get("recorrido")
if rec:
    res_rec = rd.dijkstra(esc, *rec)
    if res_rec["error"]:
        aviso.warning(res_rec["error"])
    else:
        aviso.success(f"Ruta {rec[0]} → {rec[1]}: {' → '.join(res_rec['ruta'])} · latencia total "
                      f"{res_rec['latencia']:.2f} ms · {res_rec['saltos']} saltos. Vea la animación en «Simulación».")

st.markdown("""
<div class="hero">
  <span class="tag">Latencia 70 % · tráfico 30 %</span><span class="tag">Grafo · tráfico</span><span class="tag">BFS</span><span class="tag">Dijkstra</span><span class="tag">AFD · 3 estados</span><span class="tag">GNN · camino óptimo</span>
  <h1>SoftCast: estado de los equipos y camino óptimo en la red</h1>
  <p>Los nodos son los 18 equipos y las aristas los 29 enlaces por donde se comunican, con su latencia (ms) y tráfico (Mbps).
  Un recorrido BFS visita cada equipo y el AFD decide si está Normal, Anómalo o Bloqueado; Dijkstra calcula el camino
  de menor costo, con prioridad en la latencia, y la red neuronal de grafos aprende a encontrarlo.</p>
</div>""", unsafe_allow_html=True)

c = st.columns(6)
c[0].markdown(kpi("Normal", m_base["Normal"], "de 18 equipos", "ok"), unsafe_allow_html=True)
c[1].markdown(kpi("Anómalo", m_base["Anómalo"], "en observación", "alerta"), unsafe_allow_html=True)
c[2].markdown(kpi("Bloqueado", m_base["Bloqueado"], "2 sorteados al iniciar", "gris"), unsafe_allow_html=True)
c[3].markdown(kpi("Tráfico total", f"{m_base['trafico_total']:.0f}", "Mbps en la red", "info"),
              unsafe_allow_html=True)
c[4].markdown(kpi("Tráfico medio", f"{m_base['trafico_medio']:.1f}", "Mbps por camino"), unsafe_allow_html=True)
c[5].markdown(kpi("Congestionados", m_base["congestionados"], "caminos ≥ 60 Mbps", "ambar"), unsafe_allow_html=True)
st.info(f"🎲 **Configuración aleatoria de esta sesión (semilla {int(semilla)}).** "
        f"Equipos bloqueados: **{' y '.join(conf['bloqueados'])}** 🔒 · Enlaces interrumpidos: "
        f"**{', '.join(f'{u}–{v}' for u, v in conf['interrumpidos'])}** ✕ · "
        f"Equipos anómalos: **{', '.join(v for v in NODOS if base.nodes[v]['estado'] == 'Anómalo') or 'ninguno'}**. "
        "Al refrescar la página (F5) o pulsar 🎲 se genera otra configuración; los botones y selectores la conservan.")

tab_red, tab_ruta, tab_bfs, tab_sim, tab_afd, tab_gnn = st.tabs([
    "🌐 Red y tráfico", "🧭 Camino óptimo (GNN)", "🔎 Recorrido BFS", "🛠️ Simulación",
    "🧩 Autómata (AFD)", "🧠 Entrenamiento de la GNN"])

# ----------------------------------------------------------------- Red y tráfico
with tab_red:
    etiquetas = st.toggle("Mostrar etiquetas sobre las aristas", value=True)
    leyenda()
    components.html(visual.grafo_html(base, 560, etiquetas=etiquetas, contenido=contenido), height=575)
    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("#### Enlaces (tráfico, latencia y peso)")
        st.dataframe(tabla_enlaces(base).sort_values("Peso", ascending=False)
                     .style.format({"Tráfico (Mbps)": "{:.1f}", "Latencia (ms)": "{:.2f}", "Peso": "{:.3f}"})
                     .bar(subset=["Latencia (ms)"], color="#155E75", vmin=0, vmax=15)
                     .map(lambda x: {"Interrumpida": "color:#FB7185", "Inactiva": "color:#94A3B8"}.get(x, ""),
                          subset=["Estado"]),
                     hide_index=True, height=420, width="stretch")
    with der:
        st.markdown("#### Equipos (nodos)")
        st.dataframe(pd.DataFrame([{"Nodo": v, "Tipo": base.nodes[v]["tipo"], "Enlaces": base.degree(v),
                                    "CPU (%)": base.nodes[v]["cpu"],
                                    "Cadena de eventos": " ".join(base.nodes[v]["cadena"]),
                                    "Estado (AFD)": base.nodes[v]["estado"],
                                    "Motivo": base.nodes[v]["motivo_bloqueo"] or ""} for v in NODOS])
                     .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["Estado (AFD)"])
                     .format({"CPU (%)": "{:.1f}"}), hide_index=True, height=420, width="stretch")

# ----------------------------------------------------------------- Camino óptimo
with tab_ruta:
    if origen == destino:
        st.warning("Elija un origen y un destino distintos en la barra lateral.")
    else:
        r, dij, comp = comparar_rutas(base, origen, destino, modelo)
        if r["ruta"]:
            tarjeta_ruta(base, rd.resumen_ruta(base, r["ruta"]), origen, destino, "GNN (red neuronal)")
            st.caption(f"Peso estimado por la red: {r['estimado']:.3f} · peso real de la ruta: "
                       f"{rd.costo_ruta(base, r['ruta']):.3f} · Dijkstra: {dij['costo']:.3f} · inferencia en {r['ms']:.1f} ms.")
        else:
            st.error(f"**No existe una ruta disponible.** {r['motivo']}")
        leyenda()
        components.html(visual.grafo_html(base, 540, ruta=r["ruta"], contenido=contenido), height=555)
        izq, der = st.columns([3, 2], gap="large")
        with izq:
            st.markdown("#### Dijkstra frente a la GNN")
            mostrar_comparacion(comp)
            explicar_diferencia(dij, r, base)
            if r["ruta"]:
                st.markdown("#### Costo de cada enlace de la ruta de la GNN")
                filas = [{"Enlace": f"{a} → {b}", "Tráfico (Mbps)": base.edges[a, b]["trafico"],
                          "Latencia (ms)": base.edges[a, b]["latencia"],
                          "Peso (Dijkstra)": base.edges[a, b]["peso"],
                          "Costo aprendido (GNN)": r["costos_aprendidos"][(a, b)]}
                         for a, b in zip(r["ruta"], r["ruta"][1:])]
                st.dataframe(pd.DataFrame(filas).style.format({"Tráfico (Mbps)": "{:.1f}", "Latencia (ms)": "{:.2f}",
                                                               "Peso (Dijkstra)": "{:.3f}",
                                                               "Costo aprendido (GNN)": "{:.3f}"}),
                             hide_index=True, width="stretch")
                st.caption("Peso = 0,30·(tráfico/140 Mbps) + 0,70·(latencia/15 ms) + 0,2 por cada extremo Anómalo. "
                           "La GNN no recibe el peso: lo aprendió de las rutas de Dijkstra.")
        with der:
            st.markdown("#### Paso de mensajes capa por capa")
            capas = r.get("capas") or []
            if capas:
                k = st.slider("Capa de la GNN", 0, len(capas) - 1, min(3, len(capas) - 1))
                st.caption(f"Distancia estimada de cada equipo hasta {destino} después de {k} capas "
                           "(∞ = el mensaje aún no ha llegado).")
                dist = pd.DataFrame([{"Equipo": v, "Distancia estimada": d} for v, d in capas[k].items()])
                st.dataframe(dist.style.format({"Distancia estimada": lambda x: "∞" if x == float("inf") else f"{x:.3f}"}),
                             hide_index=True, height=420, width="stretch")

# ----------------------------------------------------------------- Recorrido BFS
with tab_bfs:
    orden = base.graph["orden_recorrido"]
    visitables = [o for o in orden if o["paso"] is not None]
    paso = st.slider("Paso del recorrido", 1, len(visitables), len(visitables))
    actual = visitables[paso - 1]
    d = base.nodes[actual["nodo"]]
    st.markdown(f'<div class="card">Paso <b>{paso}</b> · visitando <b>{actual["nodo"]}</b> ({d["tipo"]}) · nivel '
                f'<b>{actual["nivel"]}</b>' + (f' · desde <b>{actual["padre"]}</b>' if actual["padre"] else " · origen")
                + f'<br><span class="mono">{" ".join(d["cadena"])}</span> → AFD: '
                f'<span class="chip {d["estado_afd"]}">{d["estado_afd"]}</span>'
                + ("" if actual["expande"] else " · <b>no se exploran sus vecinos</b>") + '</div>',
                unsafe_allow_html=True)
    leyenda(recorrido=True)
    components.html(visual.grafo_html(base, 520, "recorrido", visitados=[o["nodo"] for o in visitables[:paso]],
                                      etiquetas=False), height=535)
    izq, der = st.columns([2, 3], gap="large")
    with izq:
        st.markdown("#### Algoritmo de recorrido (BFS)")
        st.code("""ALGORITMO RecorrerRed(G, M)
  Q ← cola con los servidores no bloqueados (nivel 0)
  mientras Q no esté vacía:
     u ← desencolar(Q)
     estado[u] ← δ̂(Normal, cadena(u))      ▸ AFD
     si estado[u] = Bloqueado: continuar
     para cada vecino v de u (orden fijo):
        si enlace activo y v no visitado:
           nivel[v] ← nivel[u] + 1; encolar(Q, v)
  no visitados → sin servicio → AFD
Complejidad: O(|V| + |E|)""", language=None)
    with der:
        st.markdown("#### Orden de visita")
        st.dataframe(pd.DataFrame([{"Paso": str(o["paso"]) if o["paso"] else "—", "Nodo": o["nodo"],
                                    "Nivel": "—" if o["nivel"] is None else str(o["nivel"]), "Desde": o["padre"] or "—",
                                    "AFD": o["estado"]} for o in orden])
                     .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["AFD"]),
                     hide_index=True, height=400, width="stretch")

# ----------------------------------------------------------------- Simulación
with tab_sim:
    st.markdown("#### Escenario de simulación")
    st.caption("Bloquee (BLQ) o habilite (HAB) equipos, bloquee enlaces o inyecte fallas. El tráfico se redistribuye, "
               "el BFS y el AFD reevalúan la red, el peso de cada enlace se recalcula y Dijkstra y la GNN buscan "
               "de nuevo la ruta.")
    c1, c2, c3, c4 = st.columns(4)
    c1.multiselect("🚫 Bloquear equipos (BLQ)", NODOS, key="sim_nodos", placeholder="Seleccione…")
    c2.multiselect("✂️ Bloquear enlaces", list(ENLACES_RED), key="sim_enlaces", placeholder="Seleccione…")
    c3.multiselect("⚡ Inyectar falla en", NODOS, key="sim_fallas", placeholder="Seleccione…")
    c4.multiselect("✅ Habilitar (HAB)", bloq_afd, key="sim_hab", placeholder="Equipos bloqueados…")

    def fijar(nodos=(), enlaces_=(), fallas=(), hab=()):
        st.session_state["sim_nodos"] = list(nodos)
        st.session_state["sim_enlaces"] = list(enlaces_)
        st.session_state["sim_fallas"] = list(fallas)
        st.session_state["sim_hab"] = list(hab)

    tramo = rd.ruta_optima(base, *(rec or (origen, destino))) or []
    b = st.columns(5)
    b[0].button("Bloquear servidor S1", on_click=fijar, kwargs={"nodos": ["S1"]}, width="stretch")
    b[1].button("Bloquear troncal S1 — S2", on_click=fijar,
                kwargs={"enlaces_": [k for k, e in ENLACES_RED.items() if set(e) == {"S1", "S2"}]}, width="stretch")
    b[2].button("Falla en la ruta actual", on_click=fijar,
                kwargs={"fallas": [v for v in tramo[1:-1] if esc.nodes[v]["tipo"] != "Servidor"][:1]},
                width="stretch")
    b[3].button("Habilitar bloqueados", on_click=fijar, kwargs={"hab": bloq_afd}, width="stretch")
    b[4].button("Restablecer", on_click=fijar, width="stretch")

    def delta(a, b_, fmt="{:+d}"):
        dd = a - b_
        return None if abs(dd) < 0.05 else fmt.format(dd)

    k = st.columns(6)
    k[0].metric("Normal", m_esc["Normal"], delta(m_esc["Normal"], m_base["Normal"]))
    k[1].metric("Anómalo", m_esc["Anómalo"], delta(m_esc["Anómalo"], m_base["Anómalo"]), delta_color="inverse")
    k[2].metric("Bloqueado", m_esc["Bloqueado"], delta(m_esc["Bloqueado"], m_base["Bloqueado"]), delta_color="inverse")
    k[3].metric("Con servicio", f"{m_esc['servicio_pct']:.0f} %",
                delta(m_esc["servicio_pct"], m_base["servicio_pct"], "{:+.0f} pp"))
    k[4].metric("Tráfico medio (Mbps)", f"{m_esc['trafico_medio']:.1f}",
                delta(m_esc["trafico_medio"], m_base["trafico_medio"], "{:+.1f} Mbps"), delta_color="inverse")
    k[5].metric("Congestionados", m_esc["congestionados"], delta(m_esc["congestionados"], m_base["congestionados"]),
                delta_color="inverse")

    st.markdown("#### Recorrido animado del camino óptimo")
    autoplay = st.session_state.pop("autoplay", False)
    dij_esc = rd.dijkstra(esc, *rec) if rec else None
    fallo = st.session_state.get("error_recorrido")
    if not rec and fallo:
        st.error(f"**Recorrido {fallo[0]} → {fallo[1]}: {fallo[2]}** No se muestra ninguna ruta ni animación.")
    elif not rec:
        st.info("Elija el origen y el destino en el panel lateral y pulse «▶ Iniciar recorrido» (o la tecla Enter) "
                "para calcular el camino óptimo y ver cómo avanza la información, enlace por enlace.")
    elif dij_esc["error"]:
        st.warning(f"{dij_esc['error']} No se ejecuta la animación.")
    if not rec or dij_esc["error"]:
        leyenda()
        components.html(visual.grafo_html(esc, 540, cambios=cambios, contenido=contenido), height=555)
    else:
        o, t = rec
        if (o, t) != (origen, destino):
            st.caption(f"Se muestra el recorrido solicitado de {o} a {t}. Pulse «Iniciar recorrido» para usar la "
                       f"selección actual ({origen} → {destino}).")
        r_esc, _, comp_esc = comparar_rutas(esc, o, t, modelo)
        opciones = ["Dijkstra (camino óptimo)"] + (["GNN (red neuronal)"] if r_esc["ruta"] else [])
        metodo = st.radio("Ruta que se anima", opciones, horizontal=True, key="metodo_anim")
        ruta_anim = dij_esc["ruta"] if metodo.startswith("Dijkstra") else r_esc["ruta"]
        leyenda("animacion")
        components.html(visual.grafo_animado_html(esc, ruta_anim, 520, metodo.split(" ")[0], autoplay, cambios,
                                                  contenido=contenido), height=520 + 125)
        tarjeta_ruta(esc, rd.resumen_ruta(esc, ruta_anim), o, t, metodo.split(" (")[0])
        izq, der = st.columns([3, 2], gap="large")
        with izq:
            st.markdown(f"#### Dijkstra frente a la GNN: {o} → {t}")
            mostrar_comparacion(comp_esc)
            explicar_diferencia(dij_esc, r_esc, esc)
            antes = rd.ruta_optima(base, o, t)
            if antes and antes != dij_esc["ruta"]:
                st.info("El escenario cambió el camino óptimo: en la red original era " + " → ".join(antes) + ".")
        with der:
            st.markdown("#### Tramos de la ruta animada")
            st.dataframe(pd.DataFrame([{"Paso": i + 1, "Tramo": f"{x['de']} → {x['a']}",
                                        "Llega a": x["estado_llegada"], "Latencia (ms)": x["latencia"],
                                        "Latencia acum. (ms)": x["lat_acum"], "Mbps": x["trafico"],
                                        "Peso": x["peso"], "Peso acum.": x["acumulado"]}
                                       for i, x in enumerate(visual.datos_recorrido(esc, ruta_anim))])
                         .style.format({"Mbps": "{:.1f}", "Latencia (ms)": "{:.2f}", "Latencia acum. (ms)": "{:.2f}",
                                        "Peso": "{:.3f}", "Peso acum.": "{:.3f}"})
                         .map(lambda x: COLOR_ESTADO.get(x, ""), subset=["Llega a"]),
                         hide_index=True, width="stretch")
    st.markdown("#### Equipos que cambiaron de estado")
    if cambios:
        st.dataframe(pd.DataFrame([{"Nodo": v, "Antes": base.nodes[v]["estado"], "Después": esc.nodes[v]["estado"],
                                    "Motivo": esc.nodes[v]["motivo_bloqueo"] or "AFD"} for v in NODOS if v in cambios])
                     .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["Antes", "Después"]),
                     hide_index=True, width="stretch")
    else:
        st.info("Ningún equipo cambió de estado respecto a la red original.")

# ----------------------------------------------------------------- AFD
afd = rd.AFD
with tab_afd:
    izq, der = st.columns(2, gap="large")
    with izq:
        st.markdown("#### Definición formal  M = (Q, Σ, δ, q₀, F)")
        st.markdown("- **Q** = {Normal, Anómalo, Bloqueado}\n- **Σ** = {OK, ERR, BLQ, HAB}\n"
                    "- **q₀** = Normal\n- **F** = {Normal}\n- **δ**: Q × Σ → Q, total (12 transiciones)")
        st.dataframe(pd.DataFrame([{"Símbolo": s, "Significado": au.DESCRIPCION_SIMBOLOS[s]} for s in au.ALFABETO]),
                     hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(afd.tabla()), hide_index=True, width="stretch")
        st.caption("→ inicial · * aceptación. Dos ERR seguidos bloquean el equipo; solo HAB lo saca de Bloqueado.")
        st.markdown('<div class="card mono">L(M) = ( OK + HAB + ERR·HAB*·OK + (ERR·HAB*·(ERR + BLQ) + BLQ)'
                    '·(OK + ERR + BLQ)*·HAB )*</div>', unsafe_allow_html=True)
    with der:
        st.markdown("#### Probar una cadena de eventos")
        ejemplos = {"Comunicación sana": "OK OK OK OK", "Falla aislada y recuperación": "OK ERR OK OK",
                    "Termina con una falla": "OK OK ERR", "Dos fallas seguidas (bloqueo)": "OK ERR ERR OK",
                    "Bloqueo manual": "OK OK BLQ OK", "Bloqueo y habilitación": "OK ERR ERR HAB OK"}
        ej = st.selectbox("Ejemplos", list(ejemplos))
        texto = st.text_input("Cadena (símbolos separados por espacios)", ejemplos[ej], key=f"cad_{ej}")
        cadena = au.parsear_cadena(texto)
        malos = [s for s in cadena if s not in au.ALFABETO]
        if malos:
            st.error(f"Símbolos fuera de Σ: {', '.join(malos)}")
        else:
            traza = afd.recorrer(cadena)
            p = st.slider("Paso de la simulación", 0, len(traza), len(traza)) if traza else 0
            act = traza[p - 1][2] if p else afd.inicial
            st.graphviz_chart(afd.a_dot(resaltar=act, arista=(traza[p - 1][0], traza[p - 1][2]) if p else None),
                              width="stretch")
            if p == len(traza):
                st.markdown(f'<span class="chip {act}">{"ACEPTADA" if afd.acepta(cadena) else "RECHAZADA"} · '
                            f'estado final {act}</span>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([{"Paso": i + 1, "Estado": a, "Símbolo": s, "Siguiente": b}
                                       for i, (a, s, b) in enumerate(traza)])
                         .style.map(lambda x: COLOR_ESTADO.get(x, ""), subset=["Estado", "Siguiente"]),
                         hide_index=True, width="stretch")

# ----------------------------------------------------------------- Entrenamiento
with tab_gnn:
    izq, der = st.columns([1, 1], gap="large")
    with izq:
        st.markdown("#### Cómo funciona la GNN")
        st.latex(r"\hat c_\theta(v\to u) = \mathrm{softplus}\big(w_2^\top \mathrm{ReLU}(W_1\,\phi_{vu} + b_1) + b_2\big)")
        st.latex(r"h_v^{(k+1)} = \min\Big(h_v^{(k)},\ \min_{u\in N(v)} \big[\hat c_\theta(v\to u) + h_u^{(k)}\big]\Big),\quad h_t^{(0)} = 0")
        st.markdown("- **φ(v→u)**: tráfico del enlace (Mbps), su cuadrado, latencia, si es inalámbrico, extremos Anómalos, CPU y tipo del equipo de llegada. La red **no** recibe el peso.\n"
                    "- Cada capa es un paso de **paso de mensajes**: el nodo recibe la distancia que conoce cada vecino "
                    "más el costo del enlace y conserva el mínimo (Bellman-Ford).\n"
                    "- Tras |V| − 1 capas, el vecino que dio el mínimo es el **siguiente salto**; se sigue desde el origen.")
    with der:
        st.markdown("#### Definición del entrenamiento")
        st.markdown(f"""
| Elemento | Valor |
|---|---|
| Datos | 300 escenarios aleatorios de la red fija · todos los pares origen–destino |
| Etiqueta | distancia óptima d*(v, t) calculada con **Dijkstra** sobre el atributo `peso` (maestro) |
| Pérdida | error cuadrático medio entre h_v y d*(v, t) |
| Optimizador | Adam, η = 0,01 · 300 épocas · 32 neuronas ocultas |
| División | {resultado.n_escenarios[0]} / {resultado.n_escenarios[1]} / {resultado.n_escenarios[2]} escenarios (70/15/15) |
| Selección | modelo con más rutas óptimas en validación |
""")
    st.markdown("#### Resultados en el conjunto de prueba")
    filas = []
    for nombre, mtr in (("GNN entrenada", resultado.prueba), ("GNN sin entrenar", resultado.sin_entrenar),
                        ("Menos saltos (BFS)", resultado.linea_base)):
        filas.append({"Método": nombre, "Rutas evaluadas": mtr["rutas"], "Rutas óptimas": 100 * mtr["optimas"],
                      "Sobrecosto medio (%)": 100 * mtr["sobrecosto_medio"],
                      "Sobrecosto máximo (%)": 100 * mtr["sobrecosto_max"]})
    st.dataframe(pd.DataFrame(filas).style.format({"Rutas óptimas": "{:.1f} %", "Sobrecosto medio (%)": "{:.2f}",
                                                   "Sobrecosto máximo (%)": "{:.1f}"}), hide_index=True, width="stretch")
    h = pd.DataFrame(resultado.historial)
    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=h["epoca"], y=h["perdida"], name="Pérdida entrenamiento", line=dict(color="#2DD4BF", width=2.5))
    fig.add_scatter(x=h["epoca"], y=h["perdida_val"], name="Pérdida validación", line=dict(color="#F59E0B", width=2.5, dash="dot"))
    fig.update_layout(title="Pérdida (error cuadrático medio)", xaxis_title="Época", yaxis_type="log")
    c1.plotly_chart(tema(fig), width="stretch")
    fig = go.Figure()
    fig.add_scatter(x=h["epoca"], y=100 * h["optimas_val"], name="Rutas óptimas (validación)",
                    line=dict(color="#A78BFA", width=2.5))
    fig.add_hline(y=100 * resultado.linea_base["optimas"], line_dash="dash", line_color="#94A3B8",
                  annotation_text="Menos saltos", annotation_position="bottom right")
    fig.update_layout(title="Rutas óptimas por época (%)", xaxis_title="Época", yaxis_range=[60, 100])
    c2.plotly_chart(tema(fig), width="stretch")
