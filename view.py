import tab

class Tab(tab.Tab):
    def __init__(self, uber):
        super().__init__(uber, name="View")

    def init_ui_for_mission(self):
        super().init_ui_for_mission(mode="pygame")

        self.create_input_box("Rotation Torque", "rotation_torque")
        self.create_input_box("Motor Force", "motor_force")
        self.create_input_box("Friction", "friction")
        self.create_input_box("Drag", "drag")
        
        self.separator()



def recalculate(tab: Tab):
    pass