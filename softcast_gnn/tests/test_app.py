"""Prueba de humo de la interfaz: sesión aleatoria, conservación del estado y botón «Iniciar recorrido»."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _app(semilla=None):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    if semilla is not None:
        at.number_input(key="semilla").set_value(semilla).run()
        assert not at.exception
    return at


def _bloqueados(at):
    texto = next(i.value for i in at.info if "Configuración aleatoria" in i.value)
    return texto.split("Equipos bloqueados: **")[1].split("**")[0]


def test_app_carga_las_seis_pestanas_con_dos_bloqueados():
    at = _app()
    assert len(at.tabs) == 6
    assert " y " in _bloqueados(at)                     # exactamente dos equipos bloqueados


def test_nueva_sesion_genera_otra_configuracion():
    semillas = {_app().session_state["semilla"] for _ in range(4)}
    assert len(semillas) > 1                            # cada sesión (refresco) sortea otra semilla


def test_interacciones_conservan_la_configuracion():
    at = _app()
    semilla, bloqueados = at.session_state["semilla"], _bloqueados(at)
    at.selectbox(key="destino").set_value("S2").run()
    next(b for b in at.button if "Iniciar recorrido" in b.label).click().run()
    at.radio(key="etiqueta_enlaces").set_value("Latencia (ms)").run()
    assert at.session_state["semilla"] == semilla and _bloqueados(at) == bloqueados


def test_boton_iniciar_recorrido_muestra_latencia_en_ms():
    at = _app(semilla=3)                                # semilla 3: bloqueados PC1 e IoT3
    next(b for b in at.button if "Iniciar recorrido" in b.label).click().run()
    assert not at.exception
    assert at.session_state["recorrido"] == ("IM1", "PC5")
    assert any("latencia total" in s.value and " ms" in s.value for s in at.success)


def test_validacion_origen_bloqueado():
    at = _app(semilla=3)
    at.selectbox(key="origen").set_value("PC1").run()   # PC1 está bloqueado con la semilla 3
    next(b for b in at.button if "Iniciar recorrido" in b.label).click().run()
    assert at.session_state["recorrido"] is None
    assert any("Bloqueado" in e.value for e in at.error)


def test_validacion_origen_igual_al_destino():
    at = _app(semilla=3)
    at.selectbox(key="destino").set_value("IM1").run()
    next(b for b in at.button if "Iniciar recorrido" in b.label).click().run()
    assert at.session_state["recorrido"] is None
    assert any("diferentes" in e.value for e in at.error)


def test_nueva_configuracion_aleatoria_cambia_la_semilla():
    at = _app(semilla=3)
    next(b for b in at.button if "Nueva configuración" in b.label).click().run()
    assert not at.exception
    assert at.session_state["semilla"] != 3 or _bloqueados(at) != "PC1 y IoT3"
