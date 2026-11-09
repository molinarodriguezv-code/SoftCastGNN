# SoftCast · Estado de los equipos (AFD), camino óptimo (Dijkstra) y GNN

Experiencia final de **Lenguajes y Autómatas**, Universidad de Cundinamarca, Fusagasugá (802M).
Paula Andrea Merchán Suaza · Victor Hernando Molina Rodríguez.

## Qué hace

- **Red fija**: los **18 equipos** son los nodos (2 servidores, 5 PC de escritorio, 5 portátiles,
  2 impresoras y 4 IoT) y los **29 enlaces** de comunicación son las aristas. Cada equipo tiene
  **3 o 4 enlaces**; la red es 3-conexa.
- **Atributos de cada arista**: `trafico` (Mbps), `latencia` (ms), `activa` / `estado_enlace`
  (Activa, Interrumpida o Inactiva) y `peso`, el costo que usa Dijkstra. **La latencia tiene prioridad (70 %)**:

  ```
  peso = 0,30 · (tráfico / 140 Mbps) + 0,70 · (latencia / 15 ms) + 0,2 · (n.º de extremos en estado Anómalo)
  ```

  140 Mbps y 15 ms son los máximos de los rangos de generación (troncal y wifi), de modo que ambas
  variables quedan en [0, 1] antes de combinarse. El peso se recalcula al inyectar fallas, bloquear
  equipos o enlaces y cuando el AFD cambia un estado.
- **Estados de los equipos**: Normal y Anómalo son transitables (un Anómalo solo encarece sus enlaces y
  se usa si es la mejor o la única alternativa); **Bloqueado** se excluye de las rutas.
- **Configuración aleatoria por sesión**: al abrir la aplicación o **refrescar la página** se sortea una
  semilla y con ella **exactamente dos equipos Bloqueados** (distintos; nunca los dos servidores), de uno a
  tres **enlaces interrumpidos** (sin desconectar la red) y el estado Normal/Anómalo de los demás.
  La configuración se guarda en `st.session_state`, por lo que los botones y selectores no la cambian.
  El botón «🎲 Nueva configuración aleatoria» equivale a refrescar; escribir una semilla la reproduce.
- **Dijkstra** con NetworkX: `nx.shortest_path(G_activo, origen, destino, weight="peso", method="dijkstra")`.
  El cuadro de resultados muestra origen, destino, ruta con el estado de cada equipo, **latencia total (ms)**,
  tráfico total (Mbps) y el peso (adimensional). Si no hay camino, lo informa y no anima nada.
- **Recorrido BFS** desde los servidores y **AFD de tres estados** sobre Σ = {OK, ERR, BLQ, HAB}.
- **GNN de camino óptimo** (Bellman-Ford neuronal): aprende el costo de cada enlace imitando a Dijkstra
  sobre el mismo `peso` (300 escenarios, MSE, Adam) y se compara con él en la interfaz.
- **Panel lateral**: origen, destino y botón **«Iniciar recorrido»** (también con la tecla **Enter**).
- **Pestaña «Simulación»**: bloqueos y habilitaciones manuales, fallas, interrupción de enlaces y la
  **animación del recorrido** (Iniciar, Pausar/Reanudar, Paso, Reiniciar y Ver completo).

## Estructura

```
softcast_gnn/
├── app.py               Interfaz Streamlit (6 pestañas y panel lateral)
├── entrenar_modelo.py   Entrena la GNN y guarda los pesos (≈ 35 s)
├── docs/                Evidencias de las pruebas
├── modelos/             gnn_rutas.npz y gnn_rutas.json (pesos y métricas)
├── core/
│   ├── automata.py      AFD de 3 estados, minimización, generador de eventos
│   ├── datos.py         Perfiles, tráfico, latencia, atributo peso y características φ
│   ├── red.py           Topología fija, BFS, simulación, Dijkstra y ruta de menos saltos
│   ├── gnn.py           GNN Bellman-Ford neuronal: entrenamiento, evaluación y búsqueda de rutas
│   └── visual.py        Grafo interactivo con Pyvis y animación del recorrido (vis.js)
├── tests/               50 pruebas (pytest): AFD, red, peso, sesión aleatoria, Dijkstra, GNN e interfaz
├── requirements.txt     Dependencias de ejecución (Streamlit Community Cloud)
├── requirements-dev.txt Dependencias de desarrollo (pytest)
└── .streamlit/config.toml
```

## Ejecución local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q          # 50 passed
streamlit run app.py
```

Para volver a entrenar la GNN: `python entrenar_modelo.py` (reescribe `modelos/`).

## Despliegue en Streamlit Community Cloud (gratuito)

1. Cree un repositorio **público** en GitHub y suba **todo el contenido** de esta carpeta (con `app.py` en la raíz
   y la carpeta `modelos/`):
   ```bash
   git init && git add . && git commit -m "SoftCast: AFD, Dijkstra y GNN"
   git branch -M main
   git remote add origin https://github.com/<usuario>/<repositorio>.git
   git push -u origin main
   ```
   (También puede usar *Add file → Upload files* en la página del repositorio.)
2. Ingrese a <https://share.streamlit.io> con su cuenta de GitHub.
3. *Create app* → *Deploy a public app from GitHub* → repositorio, rama `main`, archivo `app.py`.
4. En *Advanced settings* elija Python 3.12 y pulse *Deploy*. La primera instalación tarda unos minutos.
5. Si la app se suspende por inactividad, ábrala y pulse *Yes, get this app back up!*.
