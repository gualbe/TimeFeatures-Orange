import re
from functools import wraps

import numpy as np
from scipy import sparse as sp

import Orange
from Orange.data import Table, Domain, StringVariable, DiscreteVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import OWWidget, Output, Msg
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin
from orangewidget.utils.signals import Input
from orangecontrib.network import Network

from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout


# ────────────────────────────  funcionaes de utilidad  ────────────────────────────
def normalize(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def calculate_weight(expr: str):
    """
    Devuelve {var_normalizada: lag}.  Solo se usa el desplazamiento de
    `shift(...)`; si no se encuentra se devuelve 1.
    """
    out = {}
    for var, lag in re.findall(r"shift\(([^,]+),\s*([-+]?\d+)\)", expr):
        out[normalize(var)] = int(lag)
    return out


# ───────────────  decorador: tabla Variable-Expression → Network  ───────────────
def from_row_col(f):
    @wraps(f)
    def wrapped(*args, data):
        data = f(*args, data)                            # tabla Variable-Expression

        # Descartar los placeholders generados por shiftBatch(...)
        rows_ok = [r for r in data if "shiftBatch(" not in str(r["Expression"])]

        variables     = [normalize(str(r[0])) for r in rows_ok]
        relations     = {v: [] for v in variables}
        variable_expr = {}
        tipo_var      = []

        pat_vars = r"\b(" + "|".join(map(re.escape, variables)) + r")\b"

        for fila in rows_ok:
            var, expr = normalize(str(fila[0])), str(fila["Expression"])

            if expr and expr.lower() != "nan":
                tipo_var.append(0)                # derivada
                variable_expr[var] = expr
            else:
                tipo_var.append(1)                # original

            for m in re.finditer(pat_vars, expr):
                dst = m.group(1)
                if dst and dst not in relations[var]:
                    relations[var].append(dst)

        # matriz dispersa con pesos = lag
        rows, cols, w = [], [], []
        for i, src in enumerate(relations):
            pesos_src = calculate_weight(variable_expr.get(src, ""))
            for dst in relations[src]:
                if dst in relations:
                    rows.append(i)
                    cols.append(list(relations).index(dst))
                    w.append(float(pesos_src.get(dst, 1)))
        
        n      = len(relations)
        w_arr  = np.asarray(w, dtype=np.float64)
        mat    = sp.csr_matrix((w_arr, (rows, cols)), shape=(n, n))

        # Network *sin* pasar nombre posicional extra
        net = Network(range(n), mat)
        net.name = f"{f.__name__}{args}"

        # etiquetas visibles = lag
        net.edge_labels = np.array([str(int(x)) if x.is_integer() else str(x)
                                    for x in w_arr],
                                   dtype=object)

        # nodos ⇒ metas (var_name, var_type)
        nombres = np.array(list(relations)).reshape(-1, 1)
        tipos   = np.array(tipo_var).reshape(-1, 1)
        return net, nombres, tipos

    return wrapped


@from_row_col
def grafo(data=None):
    return data


# ────────────────────────────  widget  ────────────────────────────
class owvardependencygraph(OWWidget, ConcurrentWidgetMixin):
    name        = "Variable Dependency Graph"
    icon        = "icons/graphgenerator.svg"
    GRAPH_TYPES = (grafo,)
    graph_type  = settings.Setting(0)

    want_main_area   = False
    resizing_enabled = False

    class Error(widget.OWWidget.Error):
        generation_error = Msg("{}")

    class Inputs:
        data = Input("Variable Definitions", Orange.data.Table)

    class Outputs:
        network = Output("Network", Network)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.data = None

        box = gui.vBox(self.controlArea, "Graph generator")
        hl  = QHBoxLayout(); box.layout().addLayout(hl)
        self.btn_generate = QPushButton("Generate")
        self.btn_generate.clicked.connect(self.generate)
        hl.addWidget(self.btn_generate)

    # ───────── entrada ─────────
    @Inputs.data
    def setData(self, data):
        self.data = data
        if data is None:
            self.Outputs.network.send(None)
            return

        attrs = [v.name for v in data.domain.attributes]
        metas = [v.name for v in data.domain.metas]
        ok    = "Variable" in attrs and "Expression" in metas
        self.btn_generate.setEnabled(ok)
        if ok:
            self.generate()
        else:
            self.Error.generation_error("Need Variable / Expression columns")
            self.Outputs.network.send(None)

    # ───────── construir grafo ─────────
    def generate(self):
        try:
            net, nombres, tipos = self.GRAPH_TYPES[self.graph_type](data=self.data)
        except Exception as e:
            self.Error.generation_error(str(e))
            self.Outputs.network.send(None)
            return

        n = len(net.nodes)
        meta_dom = Domain([], [], [
            StringVariable("var_name"),
            DiscreteVariable("var_type", values=["Derived", "Original"])
        ])
        metas = np.empty((n, 2), dtype=object)
        net.nodes = Table(meta_dom, np.zeros((n, 0)), np.zeros((n, 0)), metas)
        net.nodes[:, "var_name"] = nombres
        net.nodes[:, "var_type"] = tipos

        self.Error.clear()
        self.Outputs.network.send(net)
