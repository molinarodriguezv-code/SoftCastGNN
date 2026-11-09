"""Grafo interactivo con Pyvis (aristas con tráfico en Mbps) y animación del recorrido de una ruta."""

from __future__ import annotations

import json
import math

import networkx as nx
from pyvis.network import Network

from .datos import TRAFICO_ALTO, TRAFICO_MEDIO

COLORES = {
    "Normal": "#22C55E", "Anómalo": "#EF4444", "Bloqueado": "#64748B",
    "pendiente": "#1E293B", "actual": "#2DD4BF", "ruta": "#22D3EE", "falla": "#E879F9",
    "sin_servicio": "#F59E0B", "cambio": "#FACC15", "fondo": "#0B1220", "texto": "#E6EDF3",
    "traf_bajo": "#3B82F6", "traf_medio": "#F59E0B", "traf_alto": "#EF4444", "inactiva": "#475569",
    "interrumpida": "#F43F5E",
}
FORMAS = {"Servidor": "diamond", "PC": "dot", "Portátil": "dot", "Impresora": "square", "IoT": "triangle"}
TAMANO = {"Servidor": 34, "PC": 22, "Portátil": 18, "Impresora": 18, "IoT": 18}

# Posiciones fijas (x, y): los 16 equipos del anillo sobre una elipse y los servidores al centro.
from .red import ANILLO  # noqa: E402

POSICIONES = {"S1": (-150, 0), "S2": (150, 0)}
for _i, _v in enumerate(ANILLO):
    _t = math.radians(-90 + 22.5 * _i)
    POSICIONES[_v] = (round(430 * math.cos(_t)), round(300 * math.sin(_t)))
# Ajustes para que ningún nodo quede sobre un enlace transversal (PT1–PC2, PT3–IoT3, IoT1–PT1, PT5–IoT2, PT5–IoT4).
POSICIONES.update({"IM1": (-520, -140), "IM2": (520, -140), "PC1": (-190, -340),
                   "PC3": (-190, 345), "PC5": (190, 345), "PT5": (0, 330)})

OPCIONES = {
    "physics": {"enabled": False},
    "interaction": {"hover": True, "tooltipDelay": 80, "zoomView": True, "dragNodes": True},
    "nodes": {"font": {"face": "Inter, Segoe UI, sans-serif", "size": 14, "color": COLORES["texto"],
                       "strokeWidth": 3, "strokeColor": COLORES["fondo"]},
              "borderWidth": 2, "shadow": {"enabled": True, "size": 6, "x": 0, "y": 2}},
    "edges": {"smooth": False,
              "font": {"size": 13, "color": "#CBD5E1", "strokeWidth": 3, "strokeColor": COLORES["fondo"],
                       "align": "top"}},
}


def color_trafico(trafico: float) -> str:
    """Azul < 30 Mbps ≤ ámbar < 60 Mbps ≤ rojo."""
    if trafico < TRAFICO_MEDIO:
        return COLORES["traf_bajo"]
    return COLORES["traf_medio"] if trafico < TRAFICO_ALTO else COLORES["traf_alto"]


def ancho_trafico(trafico: float) -> float:
    """Grosor de la arista proporcional al tráfico (Mbps)."""
    return 1.5 + 7.0 * min(trafico / 100.0, 1.0)


def _tooltip(v: str, d: dict) -> str:
    lineas = [f"{v} · {d['tipo']}", d["rol"], f"Área: {d['area']}", f"Estado (AFD): {d.get('estado', '—')}"]
    if d.get("motivo_bloqueo"):
        lineas.append(f"Motivo: {d['motivo_bloqueo']}")
    lineas.append(f"CPU: {d['cpu']:.1f} %")
    if d.get("cadena") is not None:
        lineas.append("Cadena: " + " ".join(d["cadena"]))
    if d.get("nivel") is not None:
        lineas.append(f"BFS: nivel {d['nivel']}" + (f", desde {d['padre']}" if d.get("padre") else ""))
    if d.get("falla_inyectada"):
        lineas.append("⚡ Falla inyectada")
    if d.get("sin_servicio"):
        lineas.append("⚠ Sin ruta hacia un servidor")
    return "\n".join(lineas)


def grafo_html(g: nx.Graph, altura: int = 560, modo: str = "estado", ruta: list[str] | None = None,
               cambios: set[str] | None = None, visitados: list[str] | None = None,
               etiquetas: bool = True, contenido: str = "ambos") -> str:
    """modo: 'estado' (color por estado del AFD) o 'recorrido' (avance del BFS).

    contenido de la etiqueta de cada enlace activo: 'latencia' (ms), 'trafico' (Mbps) o 'ambos'.
    Enlaces: activos con color/grosor por tráfico; interrumpidos en rojo punteado;
    inactivos por un equipo bloqueado en gris punteado.
    """
    cambios = cambios or set()
    en_ruta = set(zip(ruta, ruta[1:])) | set(zip(ruta[1:], ruta)) if ruta else set()
    red = Network(height=f"{altura}px", width="100%", bgcolor=COLORES["fondo"],
                  font_color=COLORES["texto"], cdn_resources="in_line")
    for v, d in g.nodes(data=True):
        color = COLORES[d.get("estado", "Normal")]
        borde, ancho = color, 2
        if modo == "recorrido":
            if visitados is not None and v not in visitados:
                color = borde = COLORES["pendiente"]
            else:
                color = borde = COLORES[d.get("estado_afd") or "Normal"]
                if visitados and v == visitados[-1]:
                    borde, ancho = COLORES["actual"], 6
        else:
            if d.get("falla_inyectada") and d.get("estado") != "Bloqueado":
                borde, ancho = COLORES["falla"], 4
            if d.get("sin_servicio"):
                borde, ancho = COLORES["sin_servicio"], 4
            if v in cambios:
                borde, ancho = COLORES["cambio"], 6
            if ruta and v in (ruta[0], ruta[-1]):
                borde, ancho = COLORES["ruta"], 6
        x, y = POSICIONES[v]
        red.add_node(v, label=f"{v} 🔒" if d.get("estado") == "Bloqueado" else v, title=_tooltip(v, d), shape=FORMAS[d["tipo"]], size=TAMANO[d["tipo"]],
                     borderWidth=ancho, x=x, y=y, physics=False,
                     color={"background": color, "border": borde,
                            "highlight": {"background": color, "border": "#FFFFFF"},
                            "hover": {"background": color, "border": "#FFFFFF"}})
    for u, v, a in g.edges(data=True):
        activa = a["activa"]
        estado_e = a.get("estado_enlace") or ("Activa" if activa else "Inactiva")
        motivo = f" ({a['motivo_enlace']})" if a.get("motivo_enlace") else ""
        titulo = (f"{u} — {v} ({a['clase']})\nLatencia: {a['latencia']:.2f} ms\n"
                  f"Tráfico: {a['trafico']:.1f} Mbps\nPeso (Dijkstra): {a.get('peso', 0):.3f}\n"
                  f"Estado: {estado_e}{motivo}")
        dashes = False
        if (u, v) in en_ruta:
            color, ancho = COLORES["ruta"], 9
        elif estado_e == "Interrumpida":
            color, ancho, dashes = COLORES["interrumpida"], 2.5, [10, 7]
        elif not activa:
            color, ancho, dashes = COLORES["inactiva"], 1.5, [3, 6]
        else:
            color, ancho = color_trafico(a["trafico"]), ancho_trafico(a["trafico"])
        texto = None
        if etiquetas and activa:
            texto = {"latencia": f"{a['latencia']:.1f} ms", "trafico": f"{a['trafico']:.0f} Mbps"}.get(
                contenido, f"{a['latencia']:.1f} ms · {a['trafico']:.0f} Mbps")
        elif etiquetas and estado_e == "Interrumpida":
            texto = "✕"
        red.add_edge(u, v, id=f"{u}|{v}", title=titulo, width=ancho, color=color, dashes=dashes, label=texto)
    red.set_options(json.dumps(OPCIONES))
    html = red.generate_html(notebook=False).replace("border: 1px solid lightgray;", "border: none;")
    estilo = ("<style>body{margin:0;background:%s;} .card{border:none!important;background:%s!important;}"
              "div.vis-tooltip{font-family:Inter,Segoe UI,sans-serif;background:#111A2B;color:#E6EDF3;"
              "border:1px solid #2DD4BF;border-radius:8px;padding:8px 10px;white-space:pre-line;font-size:13px;}"
              "#mynetwork{border:0!important;outline:0!important;} html,body{overflow:hidden;}</style>"
              % (COLORES["fondo"], COLORES["fondo"]))
    ajuste = ("<script>(function(){var t=null;function fit(){if(window.network){network.redraw();"
              "network.fit({animation:false});}}new ResizeObserver(function(){clearTimeout(t);"
              "t=setTimeout(fit,120);}).observe(document.body);setTimeout(fit,60);})();</script>")
    html = html.replace("</body>", ajuste + "</body>", 1)
    return html.replace("</head>", estilo + "</head>", 1)


# --------------------------------------------------------------------------
# Animación del recorrido de una ruta (vis.js en el navegador)
# --------------------------------------------------------------------------
_ANIMACION_HTML = """
<div id="anim-ctl">
  <div class="fila">
    <button id="b-iniciar" class="prim">▶ Iniciar</button>
    <button id="b-pausa">⏸ Pausar</button>
    <button id="b-paso">⏭ Paso</button>
    <button id="b-reiniciar">↺ Reiniciar</button>
    <button id="b-completo">⏩ Ver completo</button>
    <label>Velocidad <select id="b-vel"><option value="1800">Lenta</option>
      <option value="1100" selected>Normal</option><option value="550">Rápida</option></select></label>
  </div>
  <div id="anim-chips"></div>
  <div id="anim-estado"></div>
</div>
<style>
#anim-ctl{font-family:Inter,Segoe UI,sans-serif;color:#E6EDF3;padding:8px 6px 6px 6px;background:#0B1220;}
#anim-ctl .fila{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}
#anim-ctl button{background:#111A2B;color:#E6EDF3;border:1px solid #2B3A55;border-radius:8px;padding:6px 12px;
  font-size:13px;cursor:pointer;} #anim-ctl button:hover{border-color:#2DD4BF;}
#anim-ctl button.prim{background:#0E7490;border-color:#22D3EE;} #anim-ctl button:disabled{opacity:.45;cursor:default;}
#anim-ctl label{font-size:13px;color:#94A3B8;margin-left:6px;} #anim-ctl select{background:#111A2B;color:#E6EDF3;
  border:1px solid #2B3A55;border-radius:6px;padding:4px;}
#anim-chips{margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;align-items:center;font-size:13px;}
#anim-chips .chip{padding:3px 9px;border-radius:999px;border:1px solid #334155;color:#94A3B8;}
#anim-chips .chip.vis{border-color:#22D3EE;color:#22D3EE;} #anim-chips .chip.anom{border-style:dashed;}
#anim-chips .chip.anom:not(.act){color:#F87171;} #anim-chips .chip.act{background:#FACC15;color:#0B1220;
  border-color:#FACC15;font-weight:700;} #anim-chips .fl{color:#475569;}
#anim-estado{margin-top:6px;font-size:13px;color:#CBD5E1;min-height:18px;}
</style>
<script>
(function(){
  const R = __DATOS__;
  const n = R.ruta.length;
  let k = 0, jugando = false, anim = null, temporizador = null, listo = false;
  const orig = {nodos: {}, aristas: {}};
  const $ = (id) => document.getElementById(id);
  function idArista(a, b){ return edges.get(a + "|" + b) ? a + "|" + b : b + "|" + a; }
  function fmt(x, d){ return x.toLocaleString("es-CO", {minimumFractionDigits: d, maximumFractionDigits: d}); }

  function pintar(){
    const nu = [], eu = [];
    R.ruta.forEach((v, i) => {
      const o = orig.nodos[v];
      let borde = o.color.border, ancho = o.borderWidth, tam = o.size;
      if (i < k - 1 || (k === n && i < n)) { borde = R.c.ruta; ancho = 6; }
      if (i === k - 1 && k < n) { borde = R.c.actual; ancho = 9; tam = o.size * 1.35; }
      if (k === n && (i === 0 || i === n - 1)) { borde = R.c.ruta; ancho = 9; tam = o.size * 1.2; }
      nu.push({id: v, size: tam, borderWidth: ancho, color: Object.assign({}, o.color, {border: borde})});
    });
    for (let i = 0; i < n - 1; i++) {
      const id = idArista(R.ruta[i], R.ruta[i + 1]), o = orig.aristas[id];
      let color = o.color, ancho = o.width;
      if (i + 1 <= k - 1) { color = R.c.ruta; ancho = 9; }
      if (anim && i === anim.i) { color = R.c.actual; ancho = 10; }
      eu.push({id: id, color: color, width: ancho});
    }
    nodes.update(nu); edges.update(eu);
    let html = "";
    R.ruta.forEach((v, i) => {
      const cls = (i === k - 1 && k < n) ? "act" : (i < k ? "vis" : "");
      const an = R.estados[i] === "Anómalo" ? " anom" : "";
      html += (i ? '<span class="fl">→</span>' : "") + '<span class="chip ' + cls + an + '" title="' + R.estados[i] + '">' + v +
              (an ? " ⚠" : "") + "</span>";
    });
    $("anim-chips").innerHTML = html;
    let txt;
    if (k === 0) txt = "Ruta " + R.metodo + " de " + R.ruta[0] + " a " + R.ruta[n - 1] + ": " + (n - 1) +
                       " saltos · latencia total " + fmt(R.latencia, 2) + " ms. Pulse ▶ Iniciar o ⏭ Paso.";
    else if (k === 1 && !anim) txt = "Paso 0 de " + (n - 1) + " · la información sale de " + R.ruta[0] + " · latencia acumulada 0 ms.";
    else if (anim) {
      const t = R.tramos[anim.i];
      txt = "Paso " + (anim.i + 1) + " de " + (n - 1) + " · la información viaja de " + t.de + " a " + t.a + ": latencia " +
            fmt(t.latencia, 2) + " ms (" + fmt(t.trafico, 1) + " Mbps, peso " + fmt(t.peso, 3) + ") · latencia acumulada al llegar " +
            fmt(t.lat_acum, 2) + " ms";
    } else {
      const t = R.tramos[Math.min(k, n) - 2];
      txt = "Paso " + (k - 1) + " de " + (n - 1) + " · llegó a " + t.a + " (" + R.estados[k - 1] + ") · latencia acumulada " +
            fmt(t.lat_acum, 2) + " ms · tráfico acumulado " + fmt(t.traf_acum, 1) + " Mbps · peso acumulado " + fmt(t.acumulado, 3);
      if (k === n) txt = "✔ Recorrido completo: " + R.ruta.join(" → ") + " · latencia total " + fmt(R.latencia, 2) +
                         " ms · tráfico total " + fmt(R.trafico, 1) + " Mbps · peso (costo del algoritmo) " + fmt(R.total, 3) +
                         " · " + (n - 1) + " saltos.";
    }
    $("anim-estado").textContent = txt;
    $("b-pausa").textContent = jugando ? "⏸ Pausar" : "▶ Reanudar";
    $("b-pausa").disabled = k === 0 || k === n;
    $("b-paso").disabled = jugando || k === n;
  }

  function dibujarPaquete(ctx){
    if (!anim) return;
    const p = network.getPositions([anim.a, anim.b]);
    const t = Math.min(1, (performance.now() - anim.t0) / anim.dur);
    const s = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const x = p[anim.a].x + (p[anim.b].x - p[anim.a].x) * s, y = p[anim.a].y + (p[anim.b].y - p[anim.a].y) * s;
    ctx.save(); ctx.shadowColor = R.c.actual; ctx.shadowBlur = 22; ctx.fillStyle = R.c.actual;
    ctx.beginPath(); ctx.arc(x, y, 11, 0, 2 * Math.PI); ctx.fill();
    ctx.shadowBlur = 0; ctx.fillStyle = "#0B1220"; ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }

  function avanzar(fin){
    if (k >= n) { if (fin) fin(); return; }
    if (k === 0) { k = 1; pintar(); if (fin) fin(); return; }
    anim = {i: k - 1, a: R.ruta[k - 1], b: R.ruta[k], t0: performance.now(), dur: +$("b-vel").value};
    pintar();
    (function cuadro(){
      network.redraw();
      if (performance.now() - anim.t0 >= anim.dur) { anim = null; k += 1; pintar(); if (fin) fin(); }
      else setTimeout(cuadro, 16);
    })();
  }

  function bucle(){
    if (!jugando) return;
    if (k >= n) { jugando = false; pintar(); return; }
    avanzar(() => { temporizador = setTimeout(bucle, 350); });
  }
  function detener(){ jugando = false; clearTimeout(temporizador); }

  function iniciar(){ detener(); anim = null; k = 0; jugando = true; pintar(); bucle(); }
  function init(){
    if (!window.network || !window.nodes || !window.edges) { setTimeout(init, 40); return; }
    nodes.get().forEach((o) => { orig.nodos[o.id] = {color: o.color, borderWidth: o.borderWidth, size: o.size}; });
    edges.get().forEach((o) => { orig.aristas[o.id] = {color: o.color, width: o.width}; });
    network.on("afterDrawing", dibujarPaquete);
    $("b-iniciar").onclick = iniciar;
    $("b-pausa").onclick = () => { if (jugando) { detener(); pintar(); } else if (!anim) { jugando = true; pintar(); bucle(); } };
    $("b-paso").onclick = () => { if (!jugando && !anim) avanzar(null); };
    $("b-reiniciar").onclick = () => { detener(); if (!anim) { k = 0; pintar(); } else { const e = setInterval(() => { if (!anim) { clearInterval(e); k = 0; pintar(); } }, 30); } };
    $("b-completo").onclick = () => { detener(); const e = setInterval(() => { if (!anim) { clearInterval(e); k = n; pintar(); } }, 30); };
    listo = true; pintar();
    if (R.autoplay) setTimeout(iniciar, 400);
  }
  init();
})();
</script>
"""


def datos_recorrido(g: nx.Graph, ruta: list[str]) -> list[dict]:
    """Tramos de la ruta con latencia, tráfico, peso y sus acumulados (lo que muestra la animación)."""
    tramos, acumulado, lat, traf = [], 0.0, 0.0, 0.0
    for a, b in zip(ruta, ruta[1:]):
        e = g.edges[a, b]
        acumulado += e["peso"]
        lat += e["latencia"]
        traf += e["trafico"]
        tramos.append({"de": a, "a": b, "estado_llegada": g.nodes[b]["estado"], "trafico": e["trafico"],
                       "latencia": e["latencia"], "peso": e["peso"], "acumulado": acumulado,
                       "lat_acum": lat, "traf_acum": traf})
    return tramos


def grafo_animado_html(g: nx.Graph, ruta: list[str], altura: int = 540, metodo: str = "Dijkstra",
                       autoplay: bool = False, cambios: set[str] | None = None, etiquetas: bool = True,
                       contenido: str = "ambos") -> str:
    """Grafo con controles para animar, paso a paso, el recorrido de la ruta real que devolvió el algoritmo."""
    html = grafo_html(g, altura, ruta=None, cambios=cambios, etiquetas=etiquetas, contenido=contenido)
    tramos = datos_recorrido(g, ruta)
    datos = {"ruta": ruta, "tramos": tramos, "metodo": metodo, "autoplay": bool(autoplay),
             "estados": [g.nodes[v]["estado"] for v in ruta],
             "total": tramos[-1]["acumulado"] if tramos else 0.0,
             "trafico": sum(t["trafico"] for t in tramos), "latencia": sum(t["latencia"] for t in tramos),
             "c": {"ruta": COLORES["ruta"], "actual": COLORES["cambio"]}}
    bloque = _ANIMACION_HTML.replace("__DATOS__", json.dumps(datos))
    i = html.index("<body>") + len("<body>")
    controles, script = bloque.split("<script>", 1)
    html = html[:i] + controles + html[i:]
    return html.replace("</body>", "<script>" + script + "</body>", 1)
