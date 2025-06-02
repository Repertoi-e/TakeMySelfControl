import tab

class Tab(tab.Tab):
    def __init__(self, uber):
        super().__init__(uber, name="View")

    def init_ui_for_mission(self):
        super().init_ui_for_mission(mode="pygame")

        self.create_input_box("RPM at 100 PWM", "rpm_at_100_pwm")
        self.create_input_box("WASD Move PWM", "wasd_move_pwm")
        self.create_input_box("WASD Turn PWM", "wasd_turn_pwm")

        self.create_input_box("Wheel Radius [cm]", "wheel_radius")
        self.create_input_box("Wheel Base [cm]", "wheel_base")
        
        self.separator()



def recalculate(tab: Tab):
    pass