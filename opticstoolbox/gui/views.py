"""pyqtgraph views: image panels (colour bar, axes, volume slider, linked zoom), a volume viewer,
curves and metrics.

Fitting rule: every image is fitted to its panel (and re-fitted when the panel is resized or new data
of a different shape arrive) until the user zooms or pans it; a double-click fits it again. Panels of
the same shape and extent share zoom: a manual zoom in one is copied to the others."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from ..core.registry import nice


def nice_labels(labs):
    return tuple(nice(x) for x in labs) if labs else labs

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

_CMAP_NAMES = {"intensity": "gray", "amplitude": "gray", "mask": "gray", "phase": "twilight", "signed": "RdBu_r",
               "depth": "viridis", "volume": "magma"}
_CMAPS: dict = {}


def colormap(kind):
    name = _CMAP_NAMES.get(kind, "gray")
    if name not in _CMAPS:
        if name == "gray":
            _CMAPS[name] = pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
        else:
            try:
                _CMAPS[name] = pg.colormap.getFromMatplotlib(name)
            except Exception:  # noqa: BLE001
                _CMAPS[name] = pg.colormap.get("viridis")
    return _CMAPS[name]


def levels_for(kind, a, fixed=None):
    if fixed is not None:
        return tuple(float(v) for v in fixed)
    f = np.asarray(a, float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return (0.0, 1.0)
    if kind == "phase":
        return (-np.pi, np.pi) if np.abs(f).max() <= np.pi + 1e-6 else (float(f.min()), float(f.max()))
    if kind == "signed":
        m = float(np.percentile(np.abs(f), 99.5))
        return (-m, m) if m > 0 else (-1.0, 1.0)
    lo, hi = np.percentile(f, [0.2, 99.8])
    if hi <= lo:
        lo, hi = float(f.min()), float(f.max())
    return (float(lo), float(hi) if hi > lo else float(lo) + 1.0)


def _rect_for(p, shape):
    """Image rectangle in view coordinates: the physical extent if given, else pixel indices."""
    if p.extent is not None:
        x0, x1, y0, y1 = p.extent
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
    return QtCore.QRectF(0, 0, shape[1], shape[0])


class _FitGLW(pg.GraphicsLayoutWidget):
    resized = QtCore.pyqtSignal()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.resized.emit()


class _Host(QtWidgets.QWidget):
    """Holds the plot widget and sizes it so the image area has the image's own proportions, centred:
    every image then fills its panel (no empty axis range around it) whatever its size or shape."""
    resized = QtCore.pyqtSignal()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.resized.emit()


class ImagePanel(QtWidgets.QFrame):
    """One image with title, colour bar, optional axes and (for volumes) a slice slider + 3-D viewer."""
    hovered = QtCore.pyqtSignal(str)
    sliceChanged = QtCore.pyqtSignal(int)
    openVolume = QtCore.pyqtSignal(object)
    openSurface = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(2)
        self.title = QtWidgets.QLabel()
        self.title.setObjectName("cardTitle")
        self.title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.title)
        self.host = _Host()
        self.host.setMinimumSize(240, 220)
        self.glw = _FitGLW(self.host)
        self.glw.ci.setContentsMargins(0, 0, 0, 0)
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.vb = self.plot.getViewBox()
        self.vb.invertY(True)
        self.vb.setAspectLocked(False)       # proportions come from sizing the view box (see _relayout)
        self.vb.setDefaultPadding(0)
        self._axes(False)
        self.img = pg.ImageItem()
        self.img.setAutoDownsample(True)        # average, do not skip, pixels when the panel is smaller than the image (no moiré on fringes)
        self.vb.addItem(self.img)
        self.cbar = None
        lay.addWidget(self.host, 1)
        self.host.resized.connect(self._relayout)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._slice)
        row.addWidget(self.slider, 1)
        self.btn3d = QtWidgets.QToolButton()
        self.btn3d.setText("3-D view")
        self.btn3d.setToolTip("Open the volume viewer (orthogonal slices, projections, linked to the other volumes)")
        self.btn3d.clicked.connect(self._open3d)
        row.addWidget(self.btn3d)
        self.volrow = QtWidgets.QWidget()
        self.volrow.setLayout(row)
        self.volrow.hide()
        lay.addWidget(self.volrow)
        self.panel = None
        self._vol = None
        self._lv = None
        self._data2d = None
        self._auto = True          # fit to the panel until the user zooms / pans
        self._syncing = False
        self.partners: list[ImagePanel] = []
        self.img.hoverEvent = self._hover
        self.glw.resized.connect(self.fit)
        self.vb.sigResized.connect(self._vb_resized)            # after the view box has its new geometry
        self._aspect = None                 # physical height / width of the image when proportions are kept
        self._lock = True
        self.vb.sigRangeChangedManually.connect(self._manual)
        self.glw.scene().sigMouseClicked.connect(self._clicked)

    # ---------------------------------------------------------------- appearance
    def set_theme(self, t):
        self.glw.setBackground(t["plot_bg"])
        for ax in ("left", "bottom"):
            a = self.plot.getAxis(ax)
            a.setPen(t["muted"])
            a.setTextPen(t["plot_fg"])

    def _axes(self, on, xlabel="", ylabel=""):
        for ax, lab in (("left", ylabel), ("bottom", xlabel)):
            self.plot.showAxis(ax, on)
            if on:
                self.plot.setLabel(ax, lab)
                self.plot.getAxis(ax).enableAutoSIPrefix(False)     # labels carry the units; no hidden ×10ⁿ
        self.plot.showAxis("top", False)
        self.plot.showAxis("right", False)

    # ---------------------------------------------------------------- data
    def reset(self):
        """Forget the previous panel (new experiment): next data are fitted afresh."""
        self.panel = None
        self._vol = None
        self._auto = True
        self.partners = []

    def set_panel(self, p):
        a = np.asarray(p.data)
        if np.iscomplexobj(a):
            a = np.abs(a)
        same = (self.panel is not None and np.shape(self.panel.data) == a.shape and self.panel.kind == p.kind
                and self.panel.extent == p.extent)
        if not same:
            self._auto = True
        self.panel = p
        self.title.setText(p.title)
        self.title.setToolTip(p.title)
        self._lock = p.lock_aspect()
        xl, yl = p.labels()
        self._axes(bool(xl or yl), xl, yl)
        if p.kind == "volume" and a.ndim == 3:
            self.slider.blockSignals(True)
            self.slider.setRange(0, a.shape[0] - 1)
            if not same:
                self.slider.setValue(a.shape[0] // 2)
            self.slider.blockSignals(False)
            self.slider.show()
            self.btn3d.setText("3-D view")
            self.btn3d.setToolTip("Open the volume viewer (orthogonal slices, projections, linked to the other volumes)")
            self.volrow.show()
            self._vol = a
            self._lv = levels_for(p.cmap or "volume", a, p.levels)
            self._show2d(a[self.slider.value()])
            self._slice_title(self.slider.value())
        else:
            self._vol = None
            self._lv = None
            if getattr(p, "surface", False) and a.ndim == 2:
                self.slider.hide()
                self.btn3d.setText("3-D surface")
                self.btn3d.setToolTip("Show this map as a rotatable 3-D surface, beside its Ground Truth")
                self.volrow.show()
            else:
                self.volrow.hide()
            self._show2d(a)
        self._relayout()

    def _open3d(self):
        if self._vol is not None:
            self.openVolume.emit(self)
        else:
            self.openSurface.emit(self)

    def _show2d(self, a):
        p = self.panel
        self._data2d = a
        if p.kind == "rgb":
            self.img.setImage(np.clip(np.nan_to_num(a), 0, 1) * 255, levels=(0, 255), autoLevels=False)
            self.img.setLookupTable(None)
            if self.cbar is not None:
                self.cbar.hide()
        else:
            lv = self._lv or levels_for(p.cmap or p.kind, a, p.levels)
            data = np.where(np.isfinite(a), a, np.nan).astype(np.float32)
            self.img.setImage(data, autoLevels=False, levels=lv)
            cm = colormap(p.cmap or p.kind)
            if self.cbar is None:
                self.cbar = pg.ColorBarItem(values=lv, colorMap=cm, interactive=True, width=12)
                self.cbar.setImageItem(self.img)
                self.glw.addItem(self.cbar, row=0, col=1)
            else:
                self.cbar.setColorMap(cm)
                self.cbar.setLevels(lv)
                self.cbar.show()
        r = _rect_for(p, a.shape)
        self.img.setRect(r)
        # zoom out no further than the whole image; zoom in freely
        self.vb.setLimits(xMin=r.left(), xMax=r.right(), yMin=r.top(), yMax=r.bottom(),
                          maxXRange=r.width(), maxYRange=r.height())
        asp = abs(r.height() / r.width()) if (self._lock and r.width()) else None
        if asp != self._aspect:
            self._aspect = asp
            self._relayout()

    # ---------------------------------------------------------------- fitting and linked zoom
    def _relayout(self, *_):
        """Size the plot widget inside its host so the view box has the image's aspect ratio (or fills the
        host when proportions need not be kept), centred. Axis and colour-bar space is measured from the
        current layout, so two passes settle it."""
        W, Hh = self.host.width(), self.host.height()
        if W < 10 or Hh < 10:
            return
        g = self.glw.size()
        vr = self.vb.geometry()
        ow = max(0.0, g.width() - vr.width()) if vr.width() > 2 else 70.0      # axis + colour-bar space
        oh = max(0.0, g.height() - vr.height()) if vr.height() > 2 else 30.0
        aw, ah = max(W - ow, 20), max(Hh - oh, 20)
        if self._aspect:
            if ah / aw > self._aspect:
                vw, vh = aw, aw * self._aspect
            else:
                vw, vh = ah / self._aspect, ah
        else:
            vw, vh = aw, ah
        w, h = min(W, int(round(vw + ow))), min(Hh, int(round(vh + oh)))
        geo = QtCore.QRect((W - w) // 2, (Hh - h) // 2, w, h)
        if geo != self.glw.geometry():
            self.glw.setGeometry(geo)          # the view box resizes later and calls back here to settle
        self.fit()

    def _vb_resized(self, *_):
        # once the layout has placed the view box, its true overhead is known: re-centre if needed
        n = getattr(self, "_settle", 0)
        if n < 4:
            self._settle = n + 1
            QtCore.QTimer.singleShot(0, self._relayout)
            QtCore.QTimer.singleShot(50, self._unsettle)
        self.fit()

    def _unsettle(self):
        self._settle = 0

    def fit(self, force=False):
        if self._data2d is None or not (self._auto or force):
            return
        r = self.img.mapRectToParent(self.img.boundingRect())
        self._syncing = True
        self.vb.setRange(rect=r, padding=0)
        self._syncing = False
        if force:
            self._auto = True

    def _manual(self, *_):
        if self._syncing:
            return
        self._auto = False
        r = self.vb.viewRect()
        for o in self.partners:
            if o is not self and o.isVisible():
                o._auto = False
                o._syncing = True
                o.vb.setRange(rect=r, padding=0)
                o._syncing = False

    def _clicked(self, ev):
        if ev.double():
            for o in [self] + self.partners:
                o.fit(force=True)

    # ---------------------------------------------------------------- volumes
    def _slice_title(self, i):
        self.title.setText(f"{self.panel.title}  [{i + 1}/{self._vol.shape[0]}]")

    def _slice(self, i):
        if self._vol is not None:
            self._show2d(self._vol[i])
            self._slice_title(i)
            self.sliceChanged.emit(i)

    def set_slice(self, i):
        if self._vol is not None:
            i = int(np.clip(i, 0, self._vol.shape[0] - 1))
            if i != self.slider.value():
                self.slider.blockSignals(True)
                self.slider.setValue(i)
                self.slider.blockSignals(False)
                self._show2d(self._vol[i])
                self._slice_title(i)

    # ---------------------------------------------------------------- readout
    def _hover(self, ev):
        if ev.isExit() or self._data2d is None:
            self.hovered.emit("")
            return
        a = self._data2d
        pos = self.img.mapToParent(ev.pos())
        r = _rect_for(self.panel, a.shape)
        j = int(np.clip((pos.x() - r.left()) / r.width() * a.shape[1], 0, a.shape[1] - 1))
        i = int(np.clip((pos.y() - r.top()) / r.height() * a.shape[0], 0, a.shape[0] - 1))
        v = a[i, j]
        val = ", ".join(f"{c:.4g}" for c in v) if np.ndim(v) else f"{v:.5g}"
        where = f"({j}, {i})"
        if self.panel.extent is not None:
            where += f" = ({pos.x():.4g}, {pos.y():.4g})"
        self.hovered.emit(f"{self.panel.title}  ·  {where}  ·  {val}")


class ImageGrid(QtWidgets.QWidget):
    """Grid of ImagePanels, reused between updates so zoom survives live re-runs. Panels with equal
    shape and extent share zoom; volume panels can move their slice sliders together."""
    hovered = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(10, 4, 10, 0)
        self.chk_link = QtWidgets.QCheckBox("Link slice sliders (Reconstruction ↔ Ground Truth)")
        self.chk_link.setChecked(True)
        self.chk_link.setToolTip("Moving the slice slider of one volume moves the others with the same depth axis")
        bar.addWidget(self.chk_link)
        bar.addStretch(1)
        self.btn_fit = QtWidgets.QToolButton()
        self.btn_fit.setText("Fit all")
        self.btn_fit.setToolTip("Fit every image to its panel (double-click an image to fit it and its linked panels)")
        self.btn_fit.clicked.connect(self.fit_all)
        bar.addWidget(self.btn_fit)
        self.barw = QtWidgets.QWidget()
        self.barw.setLayout(bar)
        outer.addWidget(self.barw)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.inner = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.inner)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setSpacing(8)
        self.scroll.setWidget(self.inner)
        outer.addWidget(self.scroll, 1)
        self.panels: list[ImagePanel] = []
        self.names: list[str] = []
        self.current: dict = {}
        self.theme = None
        self.cols = 3
        self.viewer = None
        self.surface_viewer = None

    def set_theme(self, t):
        self.theme = t
        for p in self.panels:
            p.set_theme(t)
        if self.surface_viewer is not None:
            self.surface_viewer.set_theme(t)

    def _new_panel(self):
        p = ImagePanel()
        p.hovered.connect(self.hovered)
        p.sliceChanged.connect(lambda i, src=p: self._sync_slices(src, i))
        p.openVolume.connect(self.open_volume)
        p.openSurface.connect(self.open_surface)
        if self.theme:
            p.set_theme(self.theme)
        self.panels.append(p)
        return p

    def show_result(self, panels: dict):
        names = list(panels)
        n = len(names)
        while len(self.panels) < n:
            self._new_panel()
        relayout = names != self.names
        if relayout:
            self._layout(n)
        for k, name in enumerate(names):
            self.panels[k].set_panel(panels[name])
        if relayout:
            for p in self.panels[:n]:
                p.partners = []
            groups = {}
            for k, name in enumerate(names):
                pd = panels[name]
                if pd.kind == "rgb" or np.ndim(pd.data) not in (2, 3):
                    continue
                shp = np.shape(pd.data)[-2:] if pd.kind == "volume" else np.shape(pd.data)[:2]
                groups.setdefault((shp, pd.extent), []).append(self.panels[k])
            for g in groups.values():
                for p in g:
                    p.partners = [o for o in g if o is not p]
        self.names = names
        self.current = panels
        nvol = sum(1 for p in panels.values() if p.kind == "volume" and np.ndim(p.data) == 3)
        self.chk_link.setVisible(nvol >= 2)
        if self.viewer is not None and self.viewer.isVisible():
            self.viewer.set_volumes(self._volumes())
        if self.surface_viewer is not None and self.surface_viewer.isVisible():
            surfs = self._surfaces()
            if surfs:
                self.surface_viewer.set_surfaces(surfs, self.surface_viewer.current_name())
            else:
                self.surface_viewer.hide()
        QtCore.QTimer.singleShot(0, self.fit_all_auto)

    def _ncols(self, n):
        want = 1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4
        w = self.scroll.viewport().width()
        fit = max(1, int(w // 300)) if w > 50 else want        # never squeeze a panel below ~300 px
        return max(1, min(want, fit))

    def _layout(self, n):
        for p in self.panels:
            self.grid.removeWidget(p)
            p.hide()
        self.cols = self._ncols(n)
        rows = int(np.ceil(n / self.cols))
        for k in range(n):
            self.grid.addWidget(self.panels[k], k // self.cols, k % self.cols)
            self.panels[k].show()
        for c in range(4):
            self.grid.setColumnStretch(c, 1 if c < self.cols else 0)
        for r in range(max(rows, 12)):
            self.grid.setRowStretch(r, 1 if r < rows else 0)
        self._size_rows(n)

    def _size_rows(self, n=None):
        """Rows about as tall as the columns are wide (images are mostly square), so nothing is shrunk
        to a thumbnail; the scroll area scrolls when that does not fit."""
        n = n or len(self.names)
        rows = int(np.ceil(max(n, 1) / max(self.cols, 1)))
        vw, vh = self.scroll.viewport().width(), self.scroll.viewport().height()
        rowh = max(300, min(0.9 * vw / max(self.cols, 1), vh - 12))
        need = int(rows * rowh)
        self.inner.setMinimumHeight(need if need > vh else 0)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        n = len(self.names)
        if n and self._ncols(n) != self.cols:
            self._layout(n)
            QtCore.QTimer.singleShot(0, self.fit_all_auto)
        elif n:
            self._size_rows()

    def fit_all_auto(self):
        for p in self.panels:
            if p.isVisible():
                p.fit()

    def fit_all(self):
        for p in self.panels:
            if p.isVisible():
                p.fit(force=True)

    def _sync_slices(self, src, i):
        if not self.chk_link.isChecked() or src._vol is None:
            return
        n = src._vol.shape[0]
        for p in self.panels:
            if p is not src and p.isVisible() and p._vol is not None:
                m = p._vol.shape[0]
                p.set_slice(i if m == n else int(round(i * (m - 1) / max(n - 1, 1))))

    def _surfaces(self):
        return {name: p for name, p in self.current.items() if getattr(p, "surface", False) and np.ndim(p.data) == 2}

    def open_surface(self, panel):
        surfs = self._surfaces()
        if not surfs:
            return
        if self.surface_viewer is None:
            self.surface_viewer = SurfaceViewer(self.window())
            if self.theme:
                self.surface_viewer.set_theme(self.theme)
        first = next((k for k, v in surfs.items() if v is panel.panel), None)
        self.surface_viewer.set_surfaces(surfs, first)
        self.surface_viewer.show()
        self.surface_viewer.raise_()

    def _volumes(self):
        return {name: p for name, p in self.current.items() if p.kind == "volume" and np.ndim(p.data) == 3}

    def open_volume(self, panel):
        vols = self._volumes()
        if not vols:
            return
        if self.viewer is None:
            self.viewer = VolumeViewer(self.window())
            if self.theme:
                self.viewer.set_theme(self.theme)
        first = next((k for k, v in vols.items() if v is panel.panel), None)
        self.viewer.set_volumes(vols, first)
        self.viewer.show()
        self.viewer.raise_()

    def clear(self):
        for p in self.panels:
            p.hide()
            p.reset()
        self.names = []
        self.current = {}
        self.chk_link.hide()
        if self.viewer is not None:
            self.viewer.hide()
        if self.surface_viewer is not None:
            self.surface_viewer.hide()


class _SliceView(pg.GraphicsLayoutWidget):
    """One orthogonal slice with a cross-hair; clicking moves the cursor."""
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, xlabel, ylabel):
        super().__init__()
        self.plot = self.addPlot()
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", ylabel)
        for ax in ("bottom", "left"):
            self.plot.getAxis(ax).enableAutoSIPrefix(False)
        self.vb = self.plot.getViewBox()
        self.vb.invertY(True)
        self.vb.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.img.setAutoDownsample(True)        # average, do not skip, pixels when the panel is smaller than the image (no moiré on fringes)
        self.vb.addItem(self.img)
        pen = pg.mkPen((0, 200, 255), width=1, style=Qt.DashLine)
        self.hl = pg.InfiniteLine(angle=0, pen=pen)
        self.vl = pg.InfiniteLine(angle=90, pen=pen)
        self.vb.addItem(self.hl)
        self.vb.addItem(self.vl)
        self.title = ""
        self.scene().sigMouseClicked.connect(self._click)
        self._fitted = None

    def set(self, a, levels, cmap, cx, cy, title, rect=None, labels=None):
        """`rect` = (x0, x1, y0, y1): physical span of the image's pixel edges; cx, cy in the same units."""
        self.img.setImage(np.nan_to_num(a).astype(np.float32), autoLevels=False, levels=levels)
        self.img.setColorMap(cmap)
        x0, x1, y0, y1 = rect if rect is not None else (0, a.shape[1], 0, a.shape[0])
        self.img.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        self.vb.setLimits(xMin=lo_x, xMax=hi_x, yMin=lo_y, yMax=hi_y, maxXRange=hi_x - lo_x, maxYRange=hi_y - lo_y)
        self.vl.setValue(cx)
        self.hl.setValue(cy)
        self.plot.setTitle(title, size="9pt")
        if labels:
            self.plot.setLabel("bottom", labels[0])
            self.plot.setLabel("left", labels[1])
        key = (a.shape, rect)
        r = (abs(y1 - y0) / max(abs(x1 - x0), 1e-30)) / (a.shape[0] / max(a.shape[1], 1))
        self.vb.setAspectLocked(bool(1 / 3 < r < 3))      # true proportions unless the axes are different quantities
        if self._fitted != key:
            self.vb.autoRange(padding=0)
            self._fitted = key

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, "vb", None) is not None:      # also called from the base constructor
            self.vb.autoRange(padding=0.02)

    def _click(self, ev):
        p = self.vb.mapSceneToView(ev.scenePos())
        self.clicked.emit(p.x(), p.y())


class VolumeViewer(QtWidgets.QDialog):
    """Orthogonal-slice volume viewer (x–y, x–z, y–z with a shared cursor), slice or maximum-intensity
    projection, one row per volume so a reconstruction and its ground truth move together; plus a 3-D
    rendering when OpenGL is available."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume viewer")
        self.resize(1250, 820)
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Show"))
        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems(["slices through the cursor", "maximum-intensity projections", "mean projections"])
        self.cb_mode.currentIndexChanged.connect(self.redraw)
        bar.addWidget(self.cb_mode)
        self.chk_shared = QtWidgets.QCheckBox("shared colour scale")
        self.chk_shared.setToolTip("Use the same colour levels for every volume (compare amplitudes)")
        self.chk_shared.toggled.connect(self.redraw)
        bar.addWidget(self.chk_shared)
        bar.addStretch(1)
        self.sliders = []
        for name in ("z", "y", "x"):
            bar.addWidget(QtWidgets.QLabel(name))
            s = QtWidgets.QSlider(Qt.Horizontal)
            s.setFixedWidth(150)
            s.valueChanged.connect(self.redraw)
            bar.addWidget(s)
            self.sliders.append(s)
        lay.addLayout(bar)
        self.tabs = QtWidgets.QTabWidget()
        self.slice_page = QtWidgets.QScrollArea()          # scrolls when several volumes are compared
        self.slice_page.setWidgetResizable(True)
        self.slice_page.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.slice_inner = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.slice_inner)
        self.slice_page.setWidget(self.slice_inner)
        self.tabs.addTab(self.slice_page, "Orthogonal slices")
        self.gl_page = None
        try:
            import pyqtgraph.opengl as gl  # noqa: F401
            import OpenGL  # noqa: F401
            self._gl = True
        except Exception:  # noqa: BLE001
            self._gl = False
        if self._gl:
            self.gl_page = QtWidgets.QWidget()
            gl_lay = QtWidgets.QVBoxLayout(self.gl_page)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Volume"))
            self.cb_gl = QtWidgets.QComboBox()
            self.cb_gl.currentIndexChanged.connect(self._render_gl)
            row.addWidget(self.cb_gl)
            row.addWidget(QtWidgets.QLabel("threshold"))
            self.sl_thr = QtWidgets.QSlider(Qt.Horizontal)
            self.sl_thr.setRange(0, 95)
            self.sl_thr.setValue(20)
            self.sl_thr.setFixedWidth(160)
            self.sl_thr.valueChanged.connect(self._render_gl)
            row.addWidget(self.sl_thr)
            row.addStretch(1)
            gl_lay.addLayout(row)
            self.glview = None
            self.gl_host = gl_lay
            self.tabs.addTab(self.gl_page, "3-D rendering")
            self.tabs.currentChanged.connect(lambda i: self._render_gl() if self.tabs.widget(i) is self.gl_page else None)
        else:
            note = QtWidgets.QLabel("3-D rendering needs PyOpenGL (pip install PyOpenGL); the orthogonal slices work without it.")
            note.setObjectName("muted")
            lay.addWidget(note)
        lay.addWidget(self.tabs, 1)
        self.views = []
        self.vols = {}
        self.theme = None

    def set_theme(self, t):
        self.theme = t

    def set_volumes(self, vols, first=None):
        names = list(vols)
        if first in names:
            names.remove(first)
            names.insert(0, first)
        self.vols = {k: vols[k] for k in names}
        shape = np.shape(self.vols[names[0]].data)
        for w in self.views:
            for v in w:
                self.grid.removeWidget(v)
                v.deleteLater()
        self.views = []
        for r, k in enumerate(names):
            row = []
            for c, (xl, yl) in enumerate((("x", "y"), ("x", "z"), ("y", "z"))):
                v = _SliceView(xl, yl)
                if self.theme:
                    v.setBackground(self.theme["plot_bg"])
                v.clicked.connect(lambda x, y, c=c: self._click(c, x, y))
                self.grid.addWidget(v, r, c)
                row.append(v)
            self.views.append(row)
        self.slice_inner.setMinimumHeight(340 * len(names) if len(names) > 2 else 0)
        for s, m in zip(self.sliders, shape):
            s.blockSignals(True)
            s.setRange(0, m - 1)
            s.setValue(m // 2)
            s.blockSignals(False)
        if self._gl:
            self.cb_gl.blockSignals(True)
            self.cb_gl.clear()
            for nm in names:
                self.cb_gl.addItem(nice(nm), nm)
            self.cb_gl.blockSignals(False)
        self.redraw()

    @staticmethod
    def _axes_of(p, shape):
        """Pixel-edge spans and labels of the (z, y, x) axes of a volume panel, in physical units when the
        panel gives them (extent / zextent), else in voxels."""
        nz, ny, nx = shape
        u = f" ({p.unit})" if p.unit else ""
        if p.extent is not None:
            x0, x1, y0, y1 = p.extent
            xs, ys = (x0, x1, p.xlabel or f"x{u}"), (y0, y1, p.ylabel or f"y{u}")
        else:
            xs, ys = (0, nx, p.xlabel or "x (px)"), (0, ny, p.ylabel or "y (px)")
        if p.zextent is not None and nz > 1:
            a, b = p.zextent
            h = 0.5 * (b - a) / (nz - 1)
            zs = (a - h, b + h, p.zlabel or "z")
        else:
            zs = (0, nz, p.zlabel or "z (slice)")
        return zs, ys, xs

    def _click(self, view, x, y):
        z0, y0, x0 = (s.value() for s in self.sliders)
        first = next(iter(self.vols.values()))
        ax = self._axes_of(first, np.shape(first.data))
        idx = lambda v, a, n: int(np.floor((v - a[0]) / (a[1] - a[0]) * n))      # physical → voxel index
        shp = np.shape(first.data)
        if view == 0:
            x0, y0 = idx(x, ax[2], shp[2]), idx(y, ax[1], shp[1])
        elif view == 1:
            x0, z0 = idx(x, ax[2], shp[2]), idx(y, ax[0], shp[0])
        else:
            y0, z0 = idx(x, ax[1], shp[1]), idx(y, ax[0], shp[0])
        for s, v in zip(self.sliders, (z0, y0, x0)):
            s.setValue(int(np.clip(v, s.minimum(), s.maximum())))

    def redraw(self, *_):
        if not self.vols:
            return
        z, y, x = (s.value() for s in self.sliders)
        mode = self.cb_mode.currentIndex()
        allv = [np.asarray(p.data, float) for p in self.vols.values()]
        shared = levels_for("volume", np.concatenate([a.ravel() for a in allv])) if self.chk_shared.isChecked() else None
        for row, (name, p), a in zip(self.views, self.vols.items(), allv):
            if a.ndim != 3:
                continue
            zz, yy, xx = (min(z, a.shape[0] - 1), min(y, a.shape[1] - 1), min(x, a.shape[2] - 1))
            (z0, z1, zl), (y0, y1, yl), (x0, x1, xl) = self._axes_of(p, a.shape)
            pz = lambda i, n, lo, hi: lo + (i + 0.5) * (hi - lo) / n          # voxel centre, physical
            Z, Y, X = pz(zz, a.shape[0], z0, z1), pz(yy, a.shape[1], y0, y1), pz(xx, a.shape[2], x0, x1)
            signed_ = (p.cmap or p.kind) in ("signed", "phase") or (np.nanmin(a) < 0 < np.nanmax(a))
            if mode == 0:
                sl = (a[zz], a[:, yy, :], a[:, :, xx])
                tags = (f"{zl.split(' (')[0]} = {Z:.3g}", f"{yl.split(' (')[0]} = {Y:.3g}", f"{xl.split(' (')[0]} = {X:.3g}")
            else:
                if mode == 1 and signed_:          # signed data: keep the value of largest magnitude
                    f = lambda v, ax: np.take_along_axis(v, np.expand_dims(np.nanargmax(np.abs(np.nan_to_num(v)), ax), ax), ax).squeeze(ax)
                else:
                    f = (lambda v, ax: np.nanmax(v, ax)) if mode == 1 else (lambda v, ax: np.nanmean(v, ax))
                sl = (f(a, 0), f(a, 1), f(a, 2))
                kind = "Max |·|" if (mode == 1 and signed_) else "MIP" if mode == 1 else "Mean"
                tags = (f"{kind} along {zl.split(' (')[0]}", f"{kind} along {yl.split(' (')[0]}", f"{kind} along {xl.split(' (')[0]}")
            lv = shared or levels_for(p.cmap or "volume", a, p.levels)
            cm = colormap(p.cmap or "volume")
            geo = (((X, Y), (x0, x1, y0, y1), (xl, yl)),
                   ((X, Z), (x0, x1, z0, z1), (xl, zl)),
                   ((Y, Z), (y0, y1, z0, z1), (yl, zl)))
            for v, s, ((cx, cy), rect, labs), tag in zip(row, sl, geo, tags):
                v.set(s, lv, cm, cx, cy, f"{nice(name)}: {tag}", rect, nice_labels(labs))
        if self.tabs.currentWidget() is self.gl_page:
            self._render_gl()

    def _render_gl(self, *_):
        if not self._gl or not self.vols:
            return
        import pyqtgraph.opengl as gl
        if self.glview is None:
            self.glview = gl.GLViewWidget()
            self.gl_host.addWidget(self.glview, 1)
        name = self.cb_gl.currentData() or next(iter(self.vols))
        a = np.nan_to_num(np.asarray(self.vols[name].data, float))
        lo, hi = levels_for("volume", a)
        v = np.clip((a - lo) / (hi - lo + 1e-30), 0, 1)
        thr = self.sl_thr.value() / 100
        cm = colormap("volume")
        rgba = cm.map(v.ravel(), mode="byte").reshape(v.shape + (4,)).copy()
        alpha = np.clip((v - thr) / (1 - thr + 1e-9), 0, 1)
        rgba[..., 3] = (alpha * 120).astype(np.ubyte)
        data = np.ascontiguousarray(np.transpose(rgba, (2, 1, 0, 3)))      # (x, y, z, 4)
        self.glview.clear()
        item = gl.GLVolumeItem(data, smooth=True, glOptions="additive")
        item.translate(-a.shape[2] / 2, -a.shape[1] / 2, -a.shape[0] / 2)
        self.glview.addItem(item)
        box = gl.GLBoxItem()
        box.setSize(a.shape[2], a.shape[1], a.shape[0])
        box.translate(-a.shape[2] / 2, -a.shape[1] / 2, -a.shape[0] / 2)
        self.glview.addItem(box)
        self.glview.setCameraPosition(distance=2.2 * max(a.shape))


def _block_reduce(a, kind, n_max=100):
    """Shrink a 2-D map to at most n_max samples a side by block averaging (NaN-aware; phase maps are
    averaged as phasors so wrapped values do not smear)."""
    a = np.asarray(a, float)
    f = int(np.ceil(max(a.shape) / n_max))
    if f <= 1:
        return a, 1
    ny, nx = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:ny, :nx]
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        if kind == "phase":
            z = np.exp(1j * b).reshape(ny // f, f, nx // f, f)
            ok = np.isfinite(b).reshape(ny // f, f, nx // f, f)
            s = np.where(ok, z, 0).sum(axis=(1, 3))
            out = np.where(ok.any(axis=(1, 3)), np.angle(s), np.nan)
        else:
            out = np.nanmean(b.reshape(ny // f, f, nx // f, f), axis=(1, 3))
    return out, f


def _unwrap_for_display(phi):
    """Unwrap a wrapped phase map for the surface view (NaN-safe); the result's median is put in (−π, π]."""
    from ..core.unwrap import unwrap
    ok = np.isfinite(phi)
    if not ok.any():
        return phi
    u = unwrap(np.where(ok, phi, 0.0), "quality-guided")
    u = np.where(ok, u, np.nan)
    m = np.nanmedian(u)
    return u - 2 * np.pi * np.round(m / (2 * np.pi))


class _SurfaceCanvas(QtWidgets.QWidget):
    """A matplotlib 3-D axes in a Qt widget: drag to rotate, wheel to zoom."""
    viewChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect("motion_notify_event", self._moved)
        self.canvas.mpl_connect("scroll_event", self._scroll)
        self._zoom = 1.0
        self.has_surface = False

    def _moved(self, ev):
        if ev.button is not None and ev.inaxes is self.ax:
            self.viewChanged.emit(float(self.ax.elev), float(self.ax.azim))

    def _scroll(self, ev):
        self._zoom = float(np.clip(self._zoom * (1.1 if ev.button == "up" else 1 / 1.1), 0.5, 4.0))
        self.ax.set_box_aspect(self._box, zoom=self._zoom * 0.9)
        self.canvas.draw_idle()

    def set_view(self, elev, azim):
        if (round(self.ax.elev, 3), round(self.ax.azim, 3)) != (round(elev, 3), round(azim, 3)):
            self.ax.view_init(elev=elev, azim=azim)
            self.canvas.draw_idle()

    def draw(self, z, x, y, title, zlabel, xlabel, ylabel, cmap, levels, box, theme):
        import matplotlib
        from matplotlib.colors import Normalize
        elev, azim = (self.ax.elev, self.ax.azim) if self.has_surface else (35.0, -60.0)
        self.ax.clear()
        self.ax.set_axis_on()
        bg = theme["plot_bg"] if theme else "#ffffff"
        fg = theme["plot_fg"] if theme else "#202020"
        self.fig.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        X, Y = np.meshgrid(x, y)
        zm = np.ma.masked_invalid(z)
        norm = Normalize(*levels)
        cols = matplotlib.colormaps[cmap](norm(np.nan_to_num(z, nan=levels[0])))
        cols[~np.isfinite(z)] = (0, 0, 0, 0)
        self.ax.plot_surface(X, Y, zm, facecolors=cols, rstride=1, cstride=1, linewidth=0,
                             antialiased=False, shade=False)
        self.ax.set_zlim(*levels)
        self.ax.set_xlim(x[0], x[-1])
        self.ax.set_ylim(y[-1], y[0])                # image orientation: y increases downwards
        for setl, lab in ((self.ax.set_xlabel, xlabel), (self.ax.set_ylabel, ylabel), (self.ax.set_zlabel, zlabel)):
            setl(lab, color=fg, fontsize=9)
        self.ax.set_title(title, color=fg, fontsize=10)
        self.ax.tick_params(colors=fg, labelsize=8)
        for a in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            a.set_pane_color((0, 0, 0, 0))
            a.line.set_color(fg)
        self._box = box
        self.ax.set_box_aspect(box, zoom=self._zoom * 0.9)
        self.ax.view_init(elev=elev, azim=azim)
        self.has_surface = True
        self.canvas.draw_idle()

    def clear_plot(self, msg=""):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text2D(0.5, 0.5, msg, transform=self.ax.transAxes, ha="center", va="center")
        self.has_surface = False
        self.canvas.draw_idle()


_MPL_CMAP = {"intensity": "gray", "amplitude": "gray", "mask": "gray", "phase": "twilight", "signed": "RdBu_r",
             "depth": "viridis", "volume": "magma"}


class SurfaceViewer(QtWidgets.QDialog):
    """Rotatable 3-D surface views of 2-D maps (reconstructed phase / height / amplitude), the chosen
    map on the left and, by default, its Ground Truth on the right; the two views rotate together."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3-D surface viewer")
        self.resize(1250, 700)
        self.theme = None
        self.surfs: dict = {}
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Left"))
        self.cb_left = QtWidgets.QComboBox()
        bar.addWidget(self.cb_left)
        bar.addWidget(QtWidgets.QLabel("Right"))
        self.cb_right = QtWidgets.QComboBox()
        bar.addWidget(self.cb_right)
        self.chk_unwrap = QtWidgets.QCheckBox("Unwrap phase")
        self.chk_unwrap.setChecked(True)
        self.chk_unwrap.setToolTip("Show wrapped phase maps unwrapped, so the surface is continuous")
        self.chk_same = QtWidgets.QCheckBox("Same z range")
        self.chk_same.setChecked(True)
        self.chk_same.setToolTip("Use one vertical scale for both surfaces (compare reconstruction and Ground Truth)")
        self.chk_link = QtWidgets.QCheckBox("Rotate together")
        self.chk_link.setChecked(True)
        for w in (self.chk_unwrap, self.chk_same, self.chk_link):
            bar.addWidget(w)
        bar.addWidget(QtWidgets.QLabel("Vertical scale"))
        self.sl_z = QtWidgets.QSlider(Qt.Horizontal)
        self.sl_z.setRange(5, 300)          # percent of the default height of the box
        self.sl_z.setValue(100)
        self.sl_z.setFixedWidth(140)
        bar.addWidget(self.sl_z)
        self.btn_reset = QtWidgets.QPushButton("Reset view")
        bar.addWidget(self.btn_reset)
        bar.addStretch(1)
        lay.addLayout(bar)
        row = QtWidgets.QHBoxLayout()
        self.left = _SurfaceCanvas()
        self.right = _SurfaceCanvas()
        row.addWidget(self.left, 1)
        row.addWidget(self.right, 1)
        lay.addLayout(row, 1)
        self.hint = QtWidgets.QLabel("Drag to rotate, wheel to zoom. Maps larger than 100 × 100 are block-averaged "
                                     "for display; masked (unreliable) pixels are left out.")
        self.hint.setObjectName("muted")
        lay.addWidget(self.hint)
        self.cb_left.currentIndexChanged.connect(self.redraw)
        self.cb_right.currentIndexChanged.connect(self.redraw)
        for w in (self.chk_unwrap, self.chk_same):
            w.toggled.connect(self.redraw)
        self.sl_z.valueChanged.connect(self.redraw)
        self.btn_reset.clicked.connect(self._reset)
        self.left.viewChanged.connect(lambda e, a: self._follow(self.right, e, a))
        self.right.viewChanged.connect(lambda e, a: self._follow(self.left, e, a))

    def set_theme(self, t):
        self.theme = t
        if self.surfs:
            self.redraw()

    def current_name(self):
        return self.cb_left.currentData()

    @staticmethod
    def partner(name, names):
        """The Ground Truth for a reconstruction (or the reconstruction for a Ground Truth), by name:
        'unwrapped phase' -> 'true phase', 'height' -> 'true height', 'true phase' -> 'phase'."""
        if name.startswith("true "):
            base = name[5:]
            for cand in names:
                if cand != name and not cand.startswith("true ") and (cand == base or cand.endswith(" " + base)):
                    return cand
            return None
        words = name.split()
        for i in range(len(words)):
            cand = "true " + " ".join(words[i:])
            if cand in names:
                return cand
        return None

    def set_surfaces(self, surfs, first=None):
        self.surfs = dict(surfs)
        names = list(self.surfs)
        first = first if first in names else names[0]
        second = self.partner(first, names)
        if second is None:          # no Ground Truth: show the companion map (amplitude beside phase, ...)
            second = next((n for n in names if n != first and not n.startswith("true ") and "error" not in n
                           and self.surfs[n].kind != self.surfs[first].kind), None)
        for cb, sel, extra in ((self.cb_left, first, False), (self.cb_right, second, True)):
            cb.blockSignals(True)
            cb.clear()
            if extra:
                cb.addItem("(none)", None)
            for n in names:
                cb.addItem(nice(self.surfs[n].title or n), n)
            i = cb.findData(sel)
            cb.setCurrentIndex(max(i, 0))
            cb.blockSignals(False)
        self.redraw()

    def _prepared(self, name):
        p = self.surfs[name]
        a = np.asarray(p.data, float)
        kind = p.cmap or p.kind
        z, f = _block_reduce(a, kind)
        wrapped = kind == "phase" and np.nanmax(np.abs(z)) <= np.pi + 1e-6
        if wrapped and self.chk_unwrap.isChecked():
            z = _unwrap_for_display(z)
            kind = "depth"
            zl = "unwrapped phase (rad)"
        else:
            zl = "phase (rad)" if kind == "phase" else (p.unit if p.unit and kind == "amplitude" else "")
        ny, nx = z.shape
        if p.extent is not None:
            x0, x1, y0, y1 = p.extent
        else:
            x0, x1, y0, y1 = 0, a.shape[1], 0, a.shape[0]
        x = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
        y = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
        xl, yl = p.labels()
        title = nice(p.title or name)
        if not zl:
            zl = title.split(",")[0] if "(" in title else ""
        return dict(z=z, x=x, y=y, kind=kind, title=title, zlabel=zl, xlabel=xl or "x", ylabel=yl or "y",
                    wx=abs(x1 - x0), wy=abs(y1 - y0))

    def redraw(self, *_):
        if not self.surfs:
            return
        ln, rn = self.cb_left.currentData(), self.cb_right.currentData()
        items = [(self.left, self._prepared(ln) if ln else None), (self.right, self._prepared(rn) if rn else None)]
        lv = {}
        for c, d in items:
            if d is not None:
                lv[id(c)] = levels_for(d["kind"], d["z"]) if d["kind"] != "phase" else (-np.pi, np.pi)
        if self.chk_same.isChecked() and len(lv) == 2:
            both = [d for _, d in items]
            if both[0]["kind"] == both[1]["kind"] or {both[0]["kind"], both[1]["kind"]} <= {"depth", "signed", "phase"}:
                lo = min(v[0] for v in lv.values())
                hi = max(v[1] for v in lv.values())
                for k in lv:
                    lv[k] = (lo, hi)
        k = self.sl_z.value() / 100.0
        for c, d in items:
            if d is None:
                c.clear_plot("(none)")
                continue
            lo, hi = lv[id(c)]
            if hi <= lo:
                hi = lo + 1.0
            m = max(d["wx"], d["wy"]) or 1.0
            box = (d["wx"] / m, d["wy"] / m, 0.45 * k)
            c.draw(d["z"], d["x"], d["y"], d["title"], d["zlabel"], d["xlabel"], d["ylabel"],
                   _MPL_CMAP.get(d["kind"], "viridis"), (lo, hi), box, self.theme)
        self._follow(self.right, self.left.ax.elev, self.left.ax.azim)

    def _follow(self, other, elev, azim):
        if self.chk_link.isChecked() and other.has_surface:
            other.set_view(elev, azim)

    def _reset(self):
        for c in (self.left, self.right):
            c._zoom = 1.0
            if c.has_surface:
                c.ax.view_init(elev=35.0, azim=-60.0)
        self.redraw()


class CurvesView(QtWidgets.QScrollArea):
    """One pyqtgraph PlotWidget per Curve."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.inner = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.inner)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setSpacing(8)
        self.setWidget(self.inner)
        self.plots: list[pg.PlotWidget] = []
        self.theme = None
        self.series = None
        self.empty = QtWidgets.QLabel("No curves for this experiment.")
        self.empty.setObjectName("muted")
        self.empty.setAlignment(Qt.AlignCenter)
        self.grid.addWidget(self.empty, 0, 0)

    def set_theme(self, t, series):
        self.theme, self.series = t, series
        for w in self.plots:
            self._style(w)

    def _style(self, w):
        if self.theme is None:
            return
        w.setBackground(self.theme["plot_bg"])
        for ax in ("left", "bottom"):
            a = w.getAxis(ax)
            a.setPen(self.theme["muted"])
            a.setTextPen(self.theme["plot_fg"])
        w.getPlotItem().titleLabel.setAttr("color", self.theme["plot_fg"])
        lg = w.getPlotItem().legend
        if lg is not None:
            lg.setLabelTextColor(self.theme["plot_fg"])

    def show_curves(self, curves):
        while len(self.plots) < len(curves):
            w = pg.PlotWidget()
            w.setMinimumHeight(280)
            w.showGrid(x=True, y=True, alpha=0.15)
            w.addLegend(offset=(-10, 10))
            self._style(w)
            self.plots.append(w)
        self.empty.setVisible(not curves)
        cols = 1 if len(curves) <= 1 else 2
        for i, w in enumerate(self.plots):
            self.grid.removeWidget(w)
            w.setVisible(i < len(curves))
        for i, c in enumerate(curves):
            w = self.plots[i]
            self.grid.addWidget(w, i // cols, i % cols)
            pi = w.getPlotItem()
            pi.clear()
            if pi.legend is not None:
                pi.legend.clear()
            pi.setTitle(c.title, size="10pt")
            pi.setLabel("bottom", c.xlabel)
            pi.setLabel("left", c.ylabel)
            for ax in ("bottom", "left"):
                pi.getAxis(ax).enableAutoSIPrefix(False)
            pi.setLogMode(x=bool(c.logx), y=bool(c.logy))
            for k, (name, (x, y)) in enumerate(c.series.items()):
                x = np.asarray(x, float)
                y = np.asarray(y, float)
                if c.logy:
                    y = np.where(y > 0, y, np.nan)
                if c.logx:
                    x = np.where(x > 0, x, np.nan)
                pal = self.series or ["#2563eb"]
                col = pal[k % len(pal)]
                if x.size <= 40 and x.size == y.size:
                    pi.plot(x, y, pen=pg.mkPen(col, width=2), symbol="o", symbolSize=5, symbolBrush=col,
                            symbolPen=None, name=name, connect="finite")
                else:
                    pi.plot(x, y, pen=pg.mkPen(col, width=2), name=name, connect="finite")
            pi.enableAutoRange()
        for r in range(len(curves) // cols + 2):
            self.grid.setRowStretch(r, 0)


class MetricsTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["metric", "value"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.verticalHeader().hide()
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setWordWrap(True)

    def show_metrics(self, metrics, runtime=None):
        items = list(metrics.items())
        if runtime is not None:
            items.append(("runtime (s)", runtime))
        self.setRowCount(len(items))
        for i, (k, v) in enumerate(items):
            self.setItem(i, 0, QtWidgets.QTableWidgetItem(nice(str(k))))
            if isinstance(v, (float, np.floating)):
                s = f"{float(v):.4g}"
            elif isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                s = str(int(v))
            else:
                s = str(v)
            it = QtWidgets.QTableWidgetItem(s)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it.setToolTip(s)
            self.setItem(i, 1, it)
        self.resizeRowsToContents()
        h = self.horizontalHeader().height() + sum(self.rowHeight(i) for i in range(len(items))) + 4
        self.setFixedHeight(min(max(h, 40), 420))
