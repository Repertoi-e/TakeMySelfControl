from PySide6.QtGui import QFont
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Tab(QFormLayout):
    def __init__(self, uber, name):
        super().__init__()
        self.uber = uber
        self.name = name

    def create_input_box(self, label, attr):
        label = QLabel(f"{label}: ")

        initial_value = getattr(self.uber.mission, attr, 0)
        setattr(self.uber.mission, attr, initial_value)  # Ensure it exists!

        line_edit = QLineEdit()
        line_edit.setText(str(initial_value))
        line_edit.setMaximumWidth(100)
        line_edit.setMinimumWidth(100)

        def on_editing_finished():
            try:
                value = float(line_edit.text())
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
        import pygame
        import threading

        def pygame_loop():
            pygame.init()
            screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("Pygame Plot")
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                screen.fill((255, 255, 255))
                pygame.draw.line(screen, (255, 0, 0), (100, 100), (500, 400))
                pygame.display.flip()
            pygame.quit()

        threading.Thread(target=pygame_loop, daemon=True).start()
