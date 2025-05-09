import tab

class Tab(tab.Tab):
    def __init__(self, uber):
        super().__init__(uber, name="View")

    def init_ui_for_mission(self):
        super().init_ui_for_mission(mode="matplotlib")

        self.create_input_box('Size [m]', "size")
        self.separator()



def recalculate(tab: Tab):
    pass