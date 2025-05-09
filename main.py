#!/usr/bin/env python

import sys
import json
import os
import importlib 
import subprocess 
import threading

os.environ["QT_API"] = "PyQt6"
os.environ["SDL_VIDEODRIVER"] = "dummy"

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QTabWidget, QMenuBar, QMenu,
                             QVBoxLayout, QHBoxLayout, QWidget, QPlainTextEdit, QMessageBox, QLabel)
from PySide6.QtGui import (QFont, QAction)
from PySide6.QtCore import QTimer

from mission import Mission
from renderer import PgRenderer, pg_surface_to_qimage

import matplotlib.pyplot as plt
plt.style.use('dark_background')

LOAD_DEFAULT = True
DEFAULT_FILE = "default_mission.json"

DONT_PROMPT_UNSAVED = True

tabs = ['view', 'config']
for m in tabs:
    globals()[m] = __import__(m)

example_mission = json.load(open(os.path.join(os.path.dirname(__file__), 'example_mission.json')))

vertical_scroll_before_clear = None

class MissionValueError(BaseException):
    def __init__(self, msg):
        self.msg = msg

class TextBoxWriter:
    def __init__(self, text_box):
        self.text_box = text_box

    def write(self, text):
        self.text_box.insertPlainText(text)
        if vertical_scroll_before_clear is not None:
            self.text_box.verticalScrollBar().setValue(vertical_scroll_before_clear)

    def flush(self):
        pass



class TakeMySelfControl(QMainWindow):
    def __init__(self):
        super().__init__()

        self.pg_renderer = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16) 

        self.tabs = [globals()[m].Tab(self) for m in tabs]

        self.mission = None
        self.unsaved_changes = False

        self.setMinimumSize(1500, 700) 

        self.show()
        self.setWindowTitle('TakeMySelfControl')
        self.setGeometry(300, 300, 800, 600)

        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)

        import platform
        if platform.system() == "Darwin":
            menuBar.setNativeMenuBar(False)

        missionMenu = QMenu("&File", self)
        menuBar.addMenu(missionMenu)
        
        action = QAction(self)
        action.setText("&New")
        action.triggered.connect(self.new_mission)
        missionMenu.addAction(action)

        action = QAction(self)
        action.setText("&Load Mission")
        action.triggered.connect(self.load_mission)
        missionMenu.addAction(action)

        action = QAction(self)
        action.setText("&Save Mission")
        action.triggered.connect(self.save_mission)
        missionMenu.addAction(action)

        if LOAD_DEFAULT:
            try:
                with open(os.path.join(os.path.dirname(__file__), DEFAULT_FILE)) as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = example_mission

            self.mission = Mission(data)
            self.init_ui_for_mission()

    def update_frame(self):
        if not self.pg_renderer or not self.mission:
            return
        
        surface = self.pg_renderer.render()
        self.pg_image = pg_surface_to_qimage(surface)

        for t in self.tabs:
            try:
                t.update_frame()
            except Exception as e:
                print(e)

    def mouse_pressed_in_game(self, x, y, drag_id=None):
        if not self.pg_renderer or not self.mission:
            return
        self.pg_renderer.mouse_pressed_in_game(x, y, drag_id)
        
    def keyPressEvent(self, event):
        if not self.pg_renderer or not self.mission:
            return
        self.pg_renderer.keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not self.pg_renderer or not self.mission:
            return
        self.pg_renderer.keyReleaseEvent(event)

    def init_ui_for_mission(self):
        self.pg_renderer = PgRenderer(self)

        self.setCentralWidget(None)
        self.tabs = [globals()[m].Tab(self) for m in tabs]

        main_layout = QHBoxLayout()

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        output_and_buttons = QVBoxLayout()

        self.console_text_box = QPlainTextEdit(self)
        
        if sys.platform == "darwin":
            font = QFont("Monaco", 10)
        else:
            font = QFont("Courier", 10)
        font.setStyleHint(QFont.Monospace)

        self.console_text_box.setFont(font)  
        self.console_text_box.setReadOnly(True)
        self.console_text_box.setLineWrapMode(QPlainTextEdit.WidgetWidth) 
        self.console_text_box.setFixedWidth(400)

        self.text_box = TextBoxWriter(self.console_text_box)

        output_and_buttons.addWidget(self.console_text_box)

        build_and_build_status = QHBoxLayout()

        build_button = QPushButton('Build', self)
        build_and_build_status.addWidget(build_button)

        build_status = QLabel(self)
        build_and_build_status.addWidget(build_status)

        output_and_buttons.addLayout(build_and_build_status)

        def build():
            self.recalculate()

            old = sys.stdout
            old_err = sys.stderr

            sys.stdout = self.text_box
            sys.stderr = self.text_box

            print("")

            build_cmd = self.mission.build_command
            dir = self.mission.directory
            print("Running command:", build_cmd)

            build_status.setText("Building...")
            build_status.setStyleSheet("color: orange;")

            def on_complete(success, output):
                if success:
                    build_status.setText("Complete.")
                    build_status.setStyleSheet("color: green;")
                else:
                    build_status.setText("Failed.")
                    build_status.setStyleSheet("color: red;")
                print(output)
                       
                sys.stdout = old
                sys.stderr = old_err

            stdout_pipe = subprocess.PIPE
            stderr_pipe = subprocess.PIPE

            def worker():
                try:
                    process = subprocess.Popen(
                        build_cmd,
                        shell=True,
                        cwd=dir,
                        stdout=stdout_pipe,
                        stderr=stderr_pipe,
                        text=True
                    )
                    stdout_lines = []
                    stderr_lines = []

                    for stdout_line in iter(process.stdout.readline, ""):
                        stdout_lines.append(stdout_line)
                        print(stdout_line, end="")  # Also print to console or UI
                    for stderr_line in iter(process.stderr.readline, ""):
                        stderr_lines.append(stderr_line)
                        print(stderr_line, end="")  # Also print to console or UI
                    
                    process.stdout.close()
                    process.stderr.close()
                    process.wait()  # Wait for the process to terminate
                    
                    success = process.returncode == 0
                    output = "".join(stdout_lines) + "".join(stderr_lines)
                except Exception as e:
                    success = False
                    output = str(e)
                on_complete(success, output)

            threading.Thread(target=worker, daemon=True).start()
         

        build_button.clicked.connect(build)
        output_and_buttons.addWidget(build_button)

        main_layout.addLayout(output_and_buttons)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        for t in self.tabs:
            try:
                t.init_ui()
            except Exception as e:
                print(e)
        
        for t in self.tabs:
            try:
                t.init_ui_for_mission()
            except Exception as e:
                print(e)
        self.recalculate()

    def recalculate(self):
        if not self.mission:
            return
        
        old = sys.stdout
        sys.stdout = self.text_box

        global vertical_scroll_before_clear
        vertical_scroll_before_clear = self.console_text_box.verticalScrollBar().value()

        self.console_text_box.clear()

        try:
            print("#######################################")
            print("#        MISSION INFORMATION          #")
            print("#######################################")
            for i, t in enumerate(self.tabs):
                print()
                m = globals()[tabs[i]]
                importlib.reload(m)
                m.recalculate(t)
                print()
                print("---------------------------------------")

        except MissionValueError as v:
            self.console_text_box.clear()
            print(v.msg)

        sys.stdout = old

    def new_mission(self):
        if self.prompt_unsaved_should_continue():
            self.mission = Mission({})
            self.init_ui_for_mission()
            self.unsaved_changes = False

    def prompt_unsaved_should_continue(self):
        if DONT_PROMPT_UNSAVED:
            return True
        
        accept = False
        if self.unsaved_changes:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                                         "You have unsaved changes. Do you want to save before exiting?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                if self.save_mission():
                    accept = True
            elif reply == QMessageBox.No:
                accept = True
        else:
            accept = True
        return accept

    def load_mission(self):
        if self.prompt_unsaved_should_continue():
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getOpenFileName(self, "Load Mission", "", "JSON Files (*.json);;All Files (*)", options=options)
            if filename:
                with open(filename, 'r') as file:
                    data = json.load(file)
                    self.mission = Mission(data)
                    self.init_ui_for_mission()
                    self.unsaved_changes = False

    def save_mission(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Mission", "", "JSON Files (*.json);;All Files (*)", options=options)
        if filename:
            with open(filename, 'w') as file:
                data = example_mission.copy()
                for key, _ in data.items():
                    if not key.startswith("__comment_"):
                        data[key] = getattr(self.mission, key)
                json.dump(data, file, indent=4)
                self.unsaved_changes = False
            return True
        else:
            return False

    def closeEvent(self, event):
        if self.prompt_unsaved_should_continue():
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TakeMySelfControl()
    sys.exit(app.exec())
