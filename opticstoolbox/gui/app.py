"""Optics Toolbox - PyQt5 / pyqtgraph desktop GUI.

    opticstoolbox-gui        or     python -m opticstoolbox.gui      (--dark for the dark theme)

Sidebar: areas. Left column: experiment (description and equations, rendered in-app from mathtext
outlines), data source (simulated scene parameters, or your own files plus the physical parameters
they need) and reconstruction parameters. Right: image panels (colour bars, linked zoom, volume
slider, hover read-out), curves and a log. Every edit re-runs in a background thread (debounced; a
running job is cancelled and restarted) when Auto is on; slow experiments start with Auto off.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from ..core import io as IO
from ..core import registry as R
from ..core.registry import nice
from .views import CurvesView, ImageGrid, MetricsTable
from .widgets import SERIES, SERIES_DARK, EquationView, ParamForm, Worker, apply_theme, card, muted

EXTRA_QSS = """
QListWidget#sidebar {{ background: {panel}; border: none; border-right: 1px solid {border}; outline: 0; padding-top: 6px; }}
QListWidget#sidebar::item {{ padding: 9px 14px; border-radius: 6px; margin: 1px 6px; color: {text}; }}
QListWidget#sidebar::item:selected {{ background: {sel}; color: {text}; font-weight: 600; }}
QListWidget#sidebar::item:hover:!selected {{ background: {bg}; }}
QLabel#brand {{ font-weight: 700; font-size: 11.5pt; padding: 10px 16px 4px 16px; background: {panel}; }}
QWidget#sidebarBox {{ background: {panel}; }}
QPushButton#seg {{ border-radius: 0; padding: 5px 14px; }}
QPushButton#seg:checked {{ background: {accent}; color: {accent_text}; border-color: {accent}; }}
"""


class FilePicker(QtWidgets.QWidget):
    """Path box + browse button for one registry Input (stack inputs accept several files or a folder)."""
    changed = QtCore.pyqtSignal()

    def __init__(self, inp, parent=None):
        super().__init__(parent)
        self.inp = inp
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("optional" if inp.optional else "required")
        self.edit.editingFinished.connect(self.changed)
        lay.addWidget(self.edit, 1)
        b = QtWidgets.QPushButton("…")
        b.setFixedWidth(34)
        b.setToolTip("Choose file(s)")
        b.clicked.connect(self.browse)
        lay.addWidget(b)
        if inp.kind == "stack":
            bf = QtWidgets.QPushButton("Folder")
            bf.setToolTip("Use every image in a folder (natural sort order)")
            bf.clicked.connect(self.browse_folder)
            lay.addWidget(bf)
        self.paths = []

    def browse(self):
        start = QtCore.QSettings("opticstoolbox", "gui").value("last_dir", "")
        if self.inp.kind == "stack":
            fs, _ = QtWidgets.QFileDialog.getOpenFileNames(self, self.inp.label, start, IO.FILE_FILTER)
        else:
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.inp.label, start, IO.FILE_FILTER)
            fs = [f] if f else []
        if fs:
            QtCore.QSettings("opticstoolbox", "gui").setValue("last_dir", os.path.dirname(fs[0]))
            self.set_paths(fs)

    def browse_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, self.inp.label)
        if d:
            self.set_paths([d])

    def set_paths(self, fs):
        self.paths = list(fs)
        self.edit.setText(self.paths[0] if len(self.paths) == 1 else f"{len(self.paths)} files: {os.path.basename(self.paths[0])} …")
        self.changed.emit()

    def value(self):
        txt = self.edit.text().strip()
        if not txt:
            return None
        if self.paths and (txt == self.paths[0] or txt.endswith(" …")):
            return self.paths if len(self.paths) > 1 else self.paths[0]
        return txt


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, theme="light"):
        super().__init__()
        R.load_all()
        self.setWindowTitle("Optics Toolbox")
        self.resize(1600, 960)
        self.settings = QtCore.QSettings("opticstoolbox", "gui")
        self.exp: R.Experiment | None = None
        self.sim_data = None
        self.sim_key = None
        self.user_arrays = None
        self.user_data = None
        self.result = None
        self.worker: Worker | None = None
        self._pending = False
        self._gen = 0
        self._progress = None
        self._t0 = 0.0
        self.debounce = QtCore.QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(220)
        self.debounce.timeout.connect(self.run)
        self.paint = QtCore.QTimer(self)
        self.paint.setInterval(80)
        self.paint.timeout.connect(self._paint_progress)
        self._build()
        self.apply_theme(theme)
        last = self.settings.value("experiment", "offaxis")
        self.select_experiment(last if last in R.EXPERIMENTS else next(iter(R.EXPERIMENTS)))

    # ================================================================== layout
    def _build(self):
        tb = self.addToolBar("main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.act_run = tb.addAction("▶  Run   F5", self.run_now)
        self.act_run.setShortcut("F5")
        self.act_stop = tb.addAction("■  Stop   Esc", self.stop)
        self.act_stop.setShortcut("Esc")
        self.act_stop.setEnabled(False)
        tb.addSeparator()
        self.act_auto = tb.addAction("Auto update")
        self.act_auto.setCheckable(True)
        self.act_auto.setToolTip("Re-run after every edit (debounced, in the background)")
        self.act_auto.toggled.connect(lambda on: self.request() if on else None)
        tb.addSeparator()
        tb.addAction("Export result…", self.export_result).setShortcut("Ctrl+E")
        tb.addAction("Save image…", self.save_image).setShortcut("Ctrl+S")
        tb.addAction("Reset parameters", self.reset_params)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.act_theme = tb.addAction("Dark theme", self.toggle_theme)
        self.act_theme.setShortcut("Ctrl+T")

        root = QtWidgets.QSplitter(Qt.Horizontal)
        root.setHandleWidth(1)
        self.setCentralWidget(root)

        # --------------------------------------------------------- sidebar
        sbox = QtWidgets.QWidget()
        sbox.setObjectName("sidebarBox")
        sl = QtWidgets.QVBoxLayout(sbox)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        brand = QtWidgets.QLabel("Optics Toolbox")
        brand.setObjectName("brand")
        sl.addWidget(brand)
        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setObjectName("sidebar")
        for area, keys in R.AREAS.items():
            it = QtWidgets.QListWidgetItem(area)
            it.setData(Qt.UserRole, area)
            it.setToolTip("\n".join(R.EXPERIMENTS[k].title for k in keys))
            self.sidebar.addItem(it)
        self.sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sidebar.currentItemChanged.connect(self._on_area)
        sl.addWidget(self.sidebar, 1)
        sbox.setMinimumWidth(250)
        sbox.setMaximumWidth(300)
        root.addWidget(sbox)

        # --------------------------------------------------------- controls column
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(420)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        scroll.setWidget(panel)
        left = QtWidgets.QVBoxLayout(panel)
        left.setContentsMargins(10, 10, 6, 10)
        left.setSpacing(10)

        f, lay = card("Experiment")
        self.cb_exp = QtWidgets.QComboBox()
        self.cb_exp.activated.connect(lambda _: self.select_experiment(self.cb_exp.currentData()))
        lay.addWidget(self.cb_exp)
        self.lbl_blurb = muted()
        lay.addWidget(self.lbl_blurb)
        self.eq = EquationView()
        lay.addWidget(self.eq)
        left.addWidget(f)

        f, lay = card("Data")
        seg = QtWidgets.QHBoxLayout()
        seg.setSpacing(0)
        self.btn_sim = QtWidgets.QPushButton("Simulated")
        self.btn_user = QtWidgets.QPushButton("My data")
        self.src_group = QtWidgets.QButtonGroup(self)
        for b in (self.btn_sim, self.btn_user):
            b.setObjectName("seg")
            b.setCheckable(True)
            self.src_group.addButton(b)
            seg.addWidget(b)
        seg.addStretch(1)
        self.btn_sim.setChecked(True)
        self.btn_sim.toggled.connect(self._on_source)
        lay.addLayout(seg)
        self.lbl_example = muted()
        lay.addWidget(self.lbl_example)
        self.src_stack = QtWidgets.QStackedWidget()
        self.scene_form = ParamForm()
        self.scene_form.changed.connect(lambda k: self.request(scene=True))
        self.src_stack.addWidget(self.scene_form)
        upage = QtWidgets.QWidget()
        ul = QtWidgets.QVBoxLayout(upage)
        ul.setContentsMargins(0, 0, 0, 0)
        self.input_form = QtWidgets.QFormLayout()
        self.input_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        ul.addLayout(self.input_form)
        self.meta_title = muted("Physical parameters of your data")
        ul.addWidget(self.meta_title)
        self.meta_form = ParamForm()
        self.meta_form.changed.connect(lambda k: self._rebuild_user(run=True))
        ul.addWidget(self.meta_form)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Load parameters (JSON)…")
        b.setToolTip('Keys may carry units, e.g. {"wl_nm": 633, "dx_um": 3.45, "z_mm": 12}')
        b.clicked.connect(self.load_meta)
        row.addWidget(b)
        row.addStretch(1)
        self.btn_load = QtWidgets.QPushButton("Load data")
        self.btn_load.setObjectName("primary")
        self.btn_load.clicked.connect(self.load_user_data)
        row.addWidget(self.btn_load)
        ul.addLayout(row)
        self.lbl_user = muted()
        ul.addWidget(self.lbl_user)
        self.src_stack.addWidget(upage)
        lay.addWidget(self.src_stack)
        left.addWidget(f)

        f, lay = card("Reconstruction")
        self.solver_form = ParamForm()
        self.solver_form.changed.connect(self._on_solver)
        lay.addWidget(self.solver_form)
        self.eq_method = EquationView(scale=0.9)
        lay.addWidget(self.eq_method)
        row = QtWidgets.QHBoxLayout()
        self.lbl_live = muted()
        row.addWidget(self.lbl_live, 1)
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.run_now)
        row.addWidget(self.btn_run)
        lay.addLayout(row)
        left.addWidget(f)

        f, lay = card("Metrics")
        self.metrics = MetricsTable()
        lay.addWidget(self.metrics)
        left.addWidget(f)
        left.addStretch(1)
        root.addWidget(scroll)

        # --------------------------------------------------------- results
        self.tabs = QtWidgets.QTabWidget()
        self.images = ImageGrid()
        self.images.hovered.connect(lambda s: self.lbl_hover.setText(s))
        self.curves = CurvesView()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.images, "Images")
        self.tabs.addTab(self.curves, "Curves")
        self.tabs.addTab(self.log, "Log")
        root.addWidget(self.tabs)
        root.setStretchFactor(2, 1)
        root.setSizes([250, 450, 1100])

        sb = self.statusBar()
        self.lbl_status = QtWidgets.QLabel("Ready")
        sb.addWidget(self.lbl_status, 1)
        self.lbl_hover = QtWidgets.QLabel()
        self.lbl_hover.setObjectName("metric")
        sb.addPermanentWidget(self.lbl_hover)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(170)
        self.progress.setTextVisible(False)
        self.progress.hide()
        sb.addPermanentWidget(self.progress)

    # ================================================================== theme
    def apply_theme(self, name):
        self.theme_name = name
        app = QtWidgets.QApplication.instance()
        t = apply_theme(app, name)
        app.setStyleSheet(app.styleSheet() + EXTRA_QSS.format(**t))
        self.t = t
        self.images.set_theme(t)
        self.curves.set_theme(t, SERIES_DARK if name == "dark" else SERIES)
        self.act_theme.setText("Light theme" if name == "dark" else "Dark theme")
        self.settings.setValue("theme", name)
        if self.result is not None:
            self.curves.show_curves(self.result.curves)
        for w in (self.eq, self.eq_method):
            w.update()

    def toggle_theme(self):
        self.apply_theme("light" if self.theme_name == "dark" else "dark")

    # ================================================================== experiment selection
    def _on_area(self, item, _prev=None):
        if item is None:
            return
        area = item.data(Qt.UserRole)
        keys = R.AREAS[area]
        self.cb_exp.blockSignals(True)
        self.cb_exp.clear()
        for k in keys:
            e = R.EXPERIMENTS[k]
            self.cb_exp.addItem(e.title, k)
        self.cb_exp.blockSignals(False)
        if self.exp is None or self.exp.key not in keys:
            self.select_experiment(keys[0])

    def select_experiment(self, key):
        if key is None:
            return
        self.stop()
        e = R.get(key)
        self.exp = e
        self.settings.setValue("experiment", key)
        # sidebar + combo (without recursion)
        for i in range(self.sidebar.count()):
            if self.sidebar.item(i).data(Qt.UserRole) == e.area:
                if self.sidebar.currentRow() != i:
                    self.sidebar.blockSignals(True)
                    self.sidebar.setCurrentRow(i)
                    self.sidebar.blockSignals(False)
                    self.cb_exp.clear()
                    for k in R.AREAS[e.area]:
                        self.cb_exp.addItem(R.EXPERIMENTS[k].title, k)
                break
        self.cb_exp.blockSignals(True)
        self.cb_exp.setCurrentIndex(max(self.cb_exp.findData(key), 0))
        self.cb_exp.blockSignals(False)
        self.lbl_blurb.setText(e.blurb)
        self.eq.set_equations(e.equations)
        self.lbl_example.setText(f"Example: {e.example}" if e.example else "")
        self.scene_form.build(e.scene, e.defaults("scene"))
        self.solver_form.build(e.solver, e.defaults("solver"))
        self._update_method_eq()
        # user-data page
        while self.input_form.rowCount():
            self.input_form.removeRow(0)
        self.pickers = {}
        for inp in e.inputs:
            fp = FilePicker(inp)
            lbl = QtWidgets.QLabel(nice(inp.label))
            lbl.setWordWrap(True)
            lbl.setToolTip(f"{inp.kind}{' (optional)' if inp.optional else ''}")
            self.input_form.addRow(lbl, fp)
            self.pickers[inp.key] = fp
        self.meta_form.build(e.meta_params(), e.defaults("scene"))
        self.meta_title.setVisible(bool(e.meta))
        self.btn_user.setEnabled(bool(e.inputs))
        self.btn_user.setToolTip("" if e.inputs else "This experiment runs on simulated data only")
        self.user_arrays = self.user_data = None
        self.lbl_user.setText("")
        self.btn_sim.blockSignals(True)
        self.btn_sim.setChecked(True)
        self.btn_sim.blockSignals(False)
        self.src_stack.setCurrentIndex(0)
        self.sim_data = self.sim_key = None
        self.result = None
        self.images.clear()
        self.curves.show_curves([])
        self.metrics.show_metrics({})
        self.act_auto.blockSignals(True)
        self.act_auto.setChecked(e.live)
        self.act_auto.blockSignals(False)
        self.lbl_live.setText("Auto update on: edits re-run in the background." if e.live
                              else "Slower experiment: press Run (or enable Auto update).")
        self.log_msg(f"{e.title}  ({e.key})")
        self.run_now()

    def _update_method_eq(self):
        m = self.solver_form.values().get("method")
        self.eq_method.set_equations(self.exp.method_equations.get(m, []) if self.exp else [])

    def _on_solver(self, key):
        if key == "method":
            self._update_method_eq()
        self.request()

    def _on_source(self, sim_checked):
        self.src_stack.setCurrentIndex(0 if sim_checked else 1)
        if sim_checked:
            self.request(now=True)
        elif self.user_data is not None:
            self.request(now=True)

    def reset_params(self):
        if self.exp is None:
            return
        self.scene_form.build(self.exp.scene, self.exp.defaults("scene"))
        self.solver_form.build(self.exp.solver, self.exp.defaults("solver"))
        self._update_method_eq()
        self.request(now=True, scene=True)

    # ================================================================== user data
    def load_meta(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Physical parameters", "", "JSON (*.json)")
        if not f:
            return
        try:
            meta = IO.read_meta(f)
        except Exception as ex:  # noqa: BLE001
            self.lbl_user.setText(f"Could not read {os.path.basename(f)}: {ex}")
            return
        n = 0
        for k, v in meta.items():
            if k in self.meta_form.widgets:
                self.meta_form.set_value(k, v)
                n += 1
        self.lbl_user.setText(f"{n} parameter(s) set from {os.path.basename(f)}")
        self._rebuild_user(run=True)

    def load_user_data(self):
        e = self.exp
        arrays = {}
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for inp in e.inputs:
                v = self.pickers[inp.key].value()
                if v is None:
                    if not inp.optional:
                        raise ValueError(f"choose a file for “{inp.label}”")
                    continue
                arrays[inp.key] = IO.read_input(v, inp.kind)
        except Exception as ex:  # noqa: BLE001
            self.lbl_user.setText(f"Could not load: {ex}")
            self.log_msg(f"load failed: {ex}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.user_arrays = arrays
        shapes = ", ".join(f"{k} {tuple(np.shape(a))}" for k, a in arrays.items())
        self.log_msg(f"loaded {shapes}")
        if self._rebuild_user(run=False):
            self.lbl_user.setText(f"Loaded: {shapes}")
            self.btn_user.setChecked(True)
            self.request(now=True)

    def _rebuild_user(self, run=True):
        if self.user_arrays is None:
            return False
        try:
            self.user_data = self.exp.make_user_data(self.user_arrays, self.meta_form.values())
        except Exception as ex:  # noqa: BLE001
            self.user_data = None
            self.lbl_user.setText(f"Data do not fit this experiment: {ex}")
            self.log_msg(f"user data rejected: {ex}")
            return False
        if run and self.btn_user.isChecked():
            self.request()
        return True

    # ================================================================== running
    def request(self, now=False, scene=False):
        if scene:
            self.sim_key = None
        if now:
            self.debounce.stop()
            self.run()
        elif self.act_auto.isChecked():
            self.debounce.start()

    def run_now(self):
        self.request(now=True)

    def run(self):
        if self.exp is None:
            return
        if self.worker is not None and self.worker.isRunning():
            self._pending = True
            self.worker.stop()
            return
        self._start()

    def _start(self):
        e = self.exp
        user = self.btn_user.isChecked()
        if user and self.user_data is None:
            self.lbl_status.setText("Load your data first (or switch to Simulated).")
            return
        scene = self.scene_form.values()
        skey = (e.key, tuple(sorted((k, str(v)) for k, v in scene.items())))
        solver = self.solver_form.values()
        data = self.user_data if user else (self.sim_data if skey == self.sim_key else None)
        self._gen += 1
        gen = self._gen

        def fn(emit, stopped):
            d = data
            fresh = None
            if d is None:
                emit((gen, -1, 0))
                d = e.make_data(scene)
                fresh = d
            last = [0.0]

            def prog(msg):
                now = time.perf_counter()
                if now - last[0] > 0.05:
                    last[0] = now
                    emit((gen, msg[0], msg[1]))
            mon = R.Monitor(on_progress=prog, stopped=stopped, preview_every=1e9)
            res = e.run(d, solver, mon)
            return gen, fresh, skey, res

        w = Worker(fn, tag=e.key)
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(self._on_done)
        w.failed.connect(self._on_failed)
        w.cancelled.connect(lambda: self.lbl_status.setText("Cancelled"))
        w.finished.connect(self._after_worker)
        self.worker = w
        self._t0 = time.perf_counter()
        self.act_stop.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.lbl_status.setText(f"Running {e.title}…")
        self.paint.start()
        w.start()

    def stop(self):
        self._pending = False
        self.debounce.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()

    def _on_progress(self, payload):
        self._progress = payload

    def _paint_progress(self):
        p, self._progress = self._progress, None
        if p is None or p[0] != self._gen:
            return
        _, i, n = p
        dt = time.perf_counter() - self._t0
        if i < 0:
            self.progress.setRange(0, 0)
            self.lbl_status.setText(f"Simulating data… {dt:.1f} s")
        elif n:
            self.progress.setRange(0, int(n))
            self.progress.setValue(int(i))
            self.lbl_status.setText(f"{self.exp.title}: {i}/{n} · {dt:.1f} s")

    def _on_done(self, out):
        self._progress = None
        gen, fresh, skey, res = out
        if fresh is not None and self.exp is not None and skey[0] == self.exp.key:
            self.sim_data, self.sim_key = fresh, skey
        if gen != self._gen:
            return
        self.result = res
        self.images.show_result(res.panels)
        self.curves.show_curves(res.curves)
        self.metrics.show_metrics(res.metrics, res.runtime)
        m = self.solver_form.values().get("method", "")
        self.lbl_status.setText(f"{self.exp.title}{' — ' + nice(m) if m else ''}: done in {res.runtime:.2f} s")
        notes = res.notes if isinstance(res.notes, (list, tuple)) else [res.notes] if res.notes else []
        for nt in notes:
            self.log_msg(nt)
        if not res.panels and res.curves:
            self.tabs.setCurrentWidget(self.curves)

    def _on_failed(self, msg):
        first = msg.splitlines()[0]
        self.lbl_status.setText("Failed: " + first)
        self.log_msg(msg)

    def _after_worker(self):
        self.paint.stop()
        self._progress = None
        self.progress.hide()
        self.act_stop.setEnabled(False)
        if self._pending:
            self._pending = False
            if self.worker is not None:
                self.worker.wait()
            self._start()

    def busy(self):
        return self.debounce.isActive() or self._pending or (self.worker is not None and self.worker.isRunning())

    # ================================================================== export
    def export_result(self):
        if self.result is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export result to folder")
        if not d:
            return
        files = IO.save_result(self.result, d, prefix=self.exp.key)
        try:
            from ..render import save_png
            m = self.solver_form.values().get("method", "")
            files.append(save_png(self.result, os.path.join(d, f"{self.exp.key}.png"), f"{self.exp.title} — {m}"))
        except Exception as ex:  # noqa: BLE001
            self.log_msg(f"PNG overview skipped: {ex}")
        self.lbl_status.setText(f"Wrote {len(files)} files to {d}")
        self.log_msg(f"exported {len(files)} files to {d}")

    def save_image(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image panels", f"{self.exp.key}.png", "PNG (*.png)")
        if f:
            self.images.inner.grab().save(f)
            self.lbl_status.setText(f"Saved {f}")

    def log_msg(self, text):
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def closeEvent(self, ev):
        self.stop()
        if self.worker is not None:
            self.worker.wait()
        super().closeEvent(ev)


def _install_excepthook():
    """PyQt aborts on exceptions escaping a slot; log them instead."""
    import traceback

    def hook(etype, value, tb):
        msg = "".join(traceback.format_exception(etype, value, tb))
        sys.stderr.write(msg)
        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, MainWindow):
                w.log_msg(msg)
                w.lbl_status.setText(f"Error: {value}")
    sys.excepthook = hook


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("Optics Toolbox")
    _install_excepthook()
    saved = QtCore.QSettings("opticstoolbox", "gui").value("theme", "light")
    theme = "dark" if "--dark" in argv else "light" if "--light" in argv else saved
    win = MainWindow(theme)
    win._app = app            # keep the application alive when main() returns (tests, notebooks)
    win.show()
    if not hasattr(sys, "ps1") and not os.environ.get("OPTICSTOOLBOX_NO_EXEC"):
        sys.exit(app.exec_())
    return win


if __name__ == "__main__":
    main()
