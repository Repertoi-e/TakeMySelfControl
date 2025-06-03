from PySide6.QtGui import (QPainter)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QTabWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPlainTextEdit,
    QLineEdit,
    QLabel,
    QMessageBox,
    QToolBar,
    QFormLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt, QRect

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class PgWidget(QWidget):
    painting: bool = False
    drag_id: int = 0

    def __init__(self, uber):
        super().__init__()
        self.uber = uber

    def widget_to_image_coords(self, pos):
        if not hasattr(self.uber, "pg_image"):
            return None

        image = self.uber.pg_image
        widget_rect = self.rect()
        image_ratio = image.width() / image.height()
        widget_ratio = widget_rect.width() / widget_rect.height()

        if widget_ratio > image_ratio:
            target_height = widget_rect.height()
            target_width = int(target_height * image_ratio)
        else:
            target_width = widget_rect.width()
            target_height = int(target_width / image_ratio)

        x = (widget_rect.width() - target_width) // 2
        y = (widget_rect.height() - target_height) // 2
        target_rect = QRect(x, y, target_width, target_height)

        if target_rect.contains(pos):
            relative_x = (pos.x() - target_rect.x()) / target_rect.width()
            relative_y = (pos.y() - target_rect.y()) / target_rect.height()
            image_x = int(relative_x * image.width())
            image_y = int(relative_y * image.height())
            return image_x, image_y
        return None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            coords = self.widget_to_image_coords(event.pos())
            if coords:
                self.drag_id += 1
                self.uber.mouse_pressed_in_game(*coords, self.drag_id)
                self.painting = True  # Flag to indicate painting has started

    def mouseMoveEvent(self, event):
        if self.painting and event.buttons() & Qt.LeftButton:
            coords = self.widget_to_image_coords(event.pos())
            if coords:
                self.uber.mouse_pressed_in_game(*coords, self.drag_id)  # Or use a dedicated "drag" function

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.painting = False

    def paintEvent(self, event):
        if hasattr(self.uber, "pg_image"):
            image = self.uber.pg_image

            painter = QPainter(self)
            widget_rect = self.rect()
            image_ratio = image.width() / image.height()
            widget_ratio = widget_rect.width() / widget_rect.height()

            # Determine scaled target rect that preserves aspect ratio
            if widget_ratio > image_ratio:
                # Widget is too wide, limit by height
                target_height = widget_rect.height()
                target_width = int(target_height * image_ratio)
            else:
                # Widget is too tall, limit by width
                target_width = widget_rect.width()
                target_height = int(target_width / image_ratio)

            x = (widget_rect.width() - target_width) // 2
            y = (widget_rect.height() - target_height) // 2
            target_rect = widget_rect.adjusted(x, y, -x, -y)

            painter.drawImage(target_rect, image)


class Tab(QFormLayout):
    def __init__(self, uber, name):
        super().__init__()
        self.uber = uber
        self.name = name

    def create_input_box(self, label, attr, max_width=100, is_float=True):
        label = QLabel(f"{label}: ")

        initial_value = getattr(self.uber.mission, attr, 0)
        setattr(self.uber.mission, attr, initial_value)  # Ensure it exists!

        line_edit = QLineEdit()
        line_edit.setText(str(initial_value))
        line_edit.setMaximumWidth(max_width)
        line_edit.setMinimumWidth(max_width)

        def on_editing_finished():
            try:
                if is_float:
                    value = float(line_edit.text())
                else:
                    value = line_edit.text()
                setattr(self.uber.mission, attr, value)
                self.uber.recalculate()
                self.uber.unsaved_changes = True
            except:
                pass

        line_edit.textChanged.connect(on_editing_finished)
        self.addRow(label, line_edit)

        return label, line_edit

    def separator(self, title=None):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setFixedHeight(2)
        self.addRow(line)

        if title is not None:
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold")
            self.addRow(label)

    def init_ui(self):
        tab = QWidget()
        self.uber.tab_widget.addTab(tab, self.name)

        self.tab_layout = QHBoxLayout()
        tab.setLayout(self.tab_layout)

    def init_ui_for_mission(self, plot_nrows=1, plot_ncols=1, mode="matplotlib"):
        info_tab_general_container = QWidget(self.uber)
        info_tab_general_container.setLayout(self)
        info_tab_general_container.setFixedWidth(300)
        self.tab_layout.addWidget(info_tab_general_container)

        if mode == "matplotlib":
            self._setup_matplotlib(plot_nrows, plot_ncols)
        elif mode == "pyqtgraph":
            self._setup_pyqtgraph()
        elif mode == "pygame":
            self._setup_pygame_window()

    def _setup_matplotlib(self, plot_nrows, plot_ncols):
        canvas_and_toolbar = QVBoxLayout()

        if plot_nrows != 1 or plot_ncols != 1:
            self.fig, self.ax = plt.subplots(
                nrows=plot_nrows, ncols=plot_ncols, figsize=(10, 8)
            )
        else:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        canvas_and_toolbar.addWidget(self.canvas)

        toolbar = NavigationToolbar(self.canvas, self.uber)
        canvas_and_toolbar.addWidget(toolbar)
        self.tab_layout.addLayout(canvas_and_toolbar)

    def _setup_pyqtgraph(self):
        import pyqtgraph as pg

        plot_widget = pg.PlotWidget()
        plot_widget.setBackground("w")
        self.ax = plot_widget.plot(pen="b")  # Example plot line
        self.tab_layout.addWidget(plot_widget)

    def _setup_pygame_window(self):
        self.pg_widget = PgWidget(self.uber)
        self.pg_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tab_layout.addWidget(self.pg_widget)

    def update_frame(self):
        if self.pg_widget:
            self.pg_widget.update()
