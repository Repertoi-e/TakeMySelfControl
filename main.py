#!/usr/bin/env python

LOAD_DEFAULT     = True

import sys
import json
import os

os.environ["QT_API"] = "PyQt6"

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QTabWidget, QMenuBar, QMenu,
                             QVBoxLayout, QHBoxLayout, QWidget, QPlainTextEdit, QLineEdit, QLabel, QMessageBox, QToolBar, QFormLayout, QSizePolicy)
from PySide6.QtGui import (QFont, QAction)

import matplotlib
import matplotlib.pyplot as plt

plt.style.use('dark_background')

import importlib

from mission import *

tabs = ['view']
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
                with open(os.path.join(os.path.dirname(__file__), 'cass_1.json')) as f:
                    data = json.load(f)
                self.mission = Mission(data)
                self.init_ui_for_mission()
            except:
                pass

    def init_ui_for_mission(self):
        self.setCentralWidget(None)
        self.tabs = [globals()[m].Tab(self) for m in tabs]

        main_layout = QHBoxLayout()

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        output_and_recalc_button = QVBoxLayout()

        self.console_text_box = QPlainTextEdit(self)
        
        # if mac:
        if sys.platform == "darwin":
            font = QFont("Monaco", 10)
        else:
            font = QFont("Courier", 10)
        font.setStyleHint(QFont.Monospace)

        self.console_text_box.setFont(font)  
        self.console_text_box.setReadOnly(True)
        self.console_text_box.setLineWrapMode(QPlainTextEdit.WidgetWidth) 
        self.console_text_box.setFixedWidth(400)
        sys.stdout = TextBoxWriter(self.console_text_box)

        output_and_recalc_button.addWidget(self.console_text_box)

        recalc_button = QPushButton('Recalculate', self)

        def recalc_clear_cache():
            setattr(self.mission, "att_map_cached", None)
            self.recalculate()

        recalc_button.clicked.connect(recalc_clear_cache)
        output_and_recalc_button.addWidget(recalc_button)

        main_layout.addLayout(output_and_recalc_button)

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
        self.recalculate()

    def recalculate(self):
        if not self.mission: return

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

    def new_mission(self):
        if self.prompt_unsaved_should_continue():
            self.mission = Mission({})
            self.init_ui_for_mission()
            self.unsaved_changes = False

    def prompt_unsaved_should_continue(self):
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
