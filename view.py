import tab

class Tab(tab.Tab):
    def __init__(self, uber):
        super().__init__(uber, name="View")

    def init_ui_for_mission(self):
        super().init_ui_for_mission(mode="pygame")

        self.create_input_box("Move RPM", "move_rpm")
        self.create_input_box("Turn RPM", "turn_rpm")

        self.create_input_box("Wheel Radius [cm]", "wheel_radius")
        self.create_input_box("Wheel Base [cm]", "wheel_base")
        self.create_input_box("Friction (mult.)", "friction")
        self.create_input_box("Drag (mult.)", "drag")
        
        self.separator()



def recalculate(tab: Tab):
    pass