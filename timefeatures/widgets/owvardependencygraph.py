import math
import re
from functools import wraps

import Orange
from Orange.widgets.widget import OWWidget
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin

import numpy as np
from scipy import sparse as sp

from Orange.data import Table, Domain, StringVariable, DiscreteVariable
from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import Output, Msg
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout

from orangecontrib.network import Network
from orangewidget.utils.signals import Input


def calculate_weight(expression):
    import re
    valores = {}

    # Funciones temporales
    patterns = {
        "shift": r'shift\(([^,]+),([-+]?\d+)\)',
        "sum": r'sum\(([^,]+),([-+]?\d+),([-+]?\d+)\)',
        "mean": r'mean\(([^,]+),([-+]?\d+),([-+]?\d+)\)',
        "count": r'count\(([^,]+),([-+]?\d+),([-+]?\d+)\)',
        "min": r'min\(([^,]+),([-+]?\d+),([-+]?\d+)\)',
        "max": r'max\(([^,]+),([-+]?\d+),([-+]?\d+)\)',
        "sd": r'sd\(([^,]+),([-+]?\d+),([-+]?\d+)\)'
    }

    for key, pattern in patterns.items():
        matches = list(re.finditer(pattern, expression))
        for match in matches:
            variable_name = match.group(1)
            if key == "shift":
                val = int(match.group(2))
            else:
                val1 = int(match.group(2))
                val2 = int(match.group(3))
                val = val1 if abs(val1) >= abs(val2) else val2
            valores[variable_name] = val

    return valores



def from_row_col(f):
    from functools import wraps

    @wraps(f)
    def wrapped(*args, data):
        import math, re, numpy as np
        from scipy import sparse as sp
        from orangecontrib.network import Network
        from Orange.data import Table, Domain, StringVariable, DiscreteVariable

        # ------------------------------------------------
        # 1. PREPARAR LISTAS DE VARIABLES Y RELACIONES
        # ------------------------------------------------
        data = f(*args, data)

        variables     = [str(row[0]).replace(" ", "_").replace("-", "_") for row in data]
        tipo_var      = []                 # 0 = derivada, 1 = original
        relaciones    = {}                 # var → [vars relacionadas]
        variable_expr = {}                 # var → expresión

        patron_vars = r'\b(' + '|'.join(map(re.escape, variables)) + r')\b'

        for fila in data:
            var = str(fila[0]).replace(" ", "_").replace("-", "_")
            expr = str(fila[1])
            if not math.isnan(fila[1]) and expr != "NaN":
                tipo_var.append(0)
                variable_expr[var] = expr
            else:
                tipo_var.append(1)

            relaciones[var] = []
            for m in re.finditer(patron_vars, expr):
                if m.group(1) and m.group(1) not in relaciones[var]:
                    relaciones[var].append(m.group(1))

        # ------------------------------------------------
        # 2. CONSTRUIR LISTAS DE ORIGEN, DESTINO Y PESO
        # ------------------------------------------------
        rows, cols, weights = [], [], []

        for i, origen in enumerate(relaciones):
            pesos_origen = calculate_weight(variable_expr.get(origen, ""))
            for destino in relaciones[origen]:
                if destino in relaciones:           # ignorar referencias ajenas
                    j = list(relaciones).index(destino)
                    rows.append(i)
                    cols.append(j)
                    peso = pesos_origen.get(destino, 1)   # 1 si no se detecta
                    weights.append(float(peso))           # ‼️ mantener signo

        # ------------------------------------------------
        # 3. CREAR EL OBJETO Network
        # ------------------------------------------------
        n              = len(relaciones)
        w_arr          = np.asarray(weights, dtype=np.float64)  # tipo correcto
        edges_sparse   = sp.csr_matrix((w_arr, (rows, cols)), shape=(n, n))
        net            = Network(range(n), edges_sparse, name=f"{f.__name__}{args}")

        # a) Asignar los pesos a la matriz (layout los usa)
        net.edges[0].edges.data = w_arr

        # b) Asignar etiquetas de arista visibles
        net.edge_labels = np.array([str(int(w)) if w.is_integer() else str(w)
                                    for w in w_arr], dtype=object)

        # ------------------------------------------------
        # 4. Crear metadatos para los nodos (nombre / tipo)
        # ------------------------------------------------
        nombres_np   = np.array(list(relaciones)).reshape(-1, 1)
        tipo_np      = np.array(tipo_var).reshape(-1, 1)   # 0 derivada, 1 original
        meta_domain  = Domain([], [], [
                              StringVariable("var_name"),
                              DiscreteVariable("var_type", values=["Derived", "Original"])
                             ])
        net.nodes = Table(meta_domain,
                          np.zeros((n, 0)), np.zeros((n, 0)),
                          np.arange(2*n).reshape(n, 2))
        net.nodes[:, "var_name"] = nombres_np
        net.nodes[:, "var_type"] = tipo_np

        return net, nombres_np, tipo_np

    return wrapped



@from_row_col
def grafo(data=None):
    return data


class owvardependencygraph(OWWidget, ConcurrentWidgetMixin):
    name = "Variable Dependency Graph"
    description = "Construct a graph with all the conexions between the variables"
    icon = "icons/graphgenerator.svg"
    keywords = "variable dependency graph, function, graph, dependency, variable"
    priority = 2240

    GRAPH_TYPES = (
        grafo,)

    graph_type = settings.Setting(0)

    want_main_area = False

    resizing_enabled = False

    settings_version = 3

    class Error(widget.OWWidget.Error):
        generation_error = Msg("{}")

    class Inputs:
        data = Input("Variable Definitions", Orange.data.Table)

    class Outputs:
        network = Output("Network", Network)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        ConcurrentWidgetMixin.__init__(self)
        self.controlArea.setMinimumWidth(360)

        self.data = None

        box = gui.vBox(self.controlArea, "Graph generator")

        toplayout = QHBoxLayout()
        toplayout.setContentsMargins(0, 0, 0, 0)
        box.layout().addLayout(toplayout)

        buttonlayout = QVBoxLayout(spacing=10)
        buttonlayout.setContentsMargins(0, 0, 0, 0)

        self.btn_generate = QPushButton(
            "Generate", toolTip="Generate dependency graph.",
            minimumWidth=10
        )
        self.btn_generate.clicked.connect(self.generate)
        self.btn_generate.setEnabled(False)
        buttonlayout.addWidget(self.btn_generate)
        toplayout.addLayout(buttonlayout, 0)

    @Inputs.data
    def setData(self, data=None):

        self.data = data

        if self.data is not None:
            if len(self.data.domain) >= 1 and (self.data.domain[0].name != "Variable" or self.data.domain[1].name != "Expression"):
                self.Error.generation_error("You need a configuration table (Variable-Expression).")
                self.Outputs.network.send(None)
            else:
                self.generate()
                self.btn_generate.setEnabled(True)
        else:
            self.Error.clear()
            self.Outputs.network.send(None)
            self.btn_generate.setEnabled(False)

    def generate(self):

        func = self.GRAPH_TYPES[self.graph_type]

        self.Error.generation_error.clear()
        try:
            network, nombres_variables, tipo_var_reshaped = func(data=self.data)
        except ValueError as exc:
            self.Error.generation_error(exc)
            network = None
        else:
            n = len(network.nodes)
            network.nodes = Table(Domain([], [], [StringVariable("var_name"), DiscreteVariable("var_type", values=["Derived", "Original"])]),
                                  np.zeros((n, 0)), np.zeros((n, 0)),
                                  np.arange(2*n).reshape((n, 2)))

            network.nodes[:, "var_name"] = nombres_variables
            network.nodes[:, "var_type"] = tipo_var_reshaped

        self.Outputs.network.send(network)
