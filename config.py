import tab

from PySide6.QtWidgets import QPushButton, QFileDialog

class Tab(tab.Tab):
    def __init__(self, uber):
        super().__init__(uber, name="Config")

    def init_ui_for_mission(self):
        super().init_ui_for_mission(mode="pygame")

        _, dir_edit = self.create_input_box("Directory", "directory", max_width=200, is_float=False)
        # add browse button:
        browse_button = QPushButton("Browse", self.uber)
        browse_button.setMaximumWidth(100)
        
        def browse_directory():
            directory = QFileDialog.getExistingDirectory(self.uber, "Select Directory")
            print(directory)
            if directory:
                dir_edit.setText(directory)
                self.uber.mission.directory = directory
                self.uber.unsaved_changes = True
                self.uber.recalculate()

        browse_button.clicked.connect(browse_directory)
        self.addRow(browse_button)

        self.create_input_box("Build cmd", "build_command", max_width=200, is_float=False)
        self.create_input_box("Run cmd", "run_command", max_width=200, is_float=False)
        
        self.separator()



def recalculate(tab: Tab):
    print("Directory:", tab.uber.mission.directory)
    print("Build cmd:", tab.uber.mission.build_command)
    print("Run cmd:", tab.uber.mission.run_command)

