"""Núcleo de SoftCast: estados de los equipos con un AFD, camino óptimo con Dijkstra y una GNN.

Módulos:
    automata  -- AFD de 3 estados (Normal, Anómalo, Bloqueado) y generador de eventos.
    datos     -- Perfiles de CPU, tráfico (Mbps), latencia (ms) y atributo 'peso' de las aristas.
    red       -- Red fija (18 equipos, 29 enlaces de grado 3–4), BFS, simulación y Dijkstra (NetworkX).
    gnn       -- GNN Bellman-Ford neuronal que aprende a encontrar el camino de menor costo.
    visual    -- Grafo interactivo con Pyvis y animación del recorrido de una ruta.
"""
