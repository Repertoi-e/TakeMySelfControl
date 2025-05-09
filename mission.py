class Mission:
    def __init__(self, data):
        self.loaded_from_data = data
        for key, value in data.items():
            if not key.startswith("__comment_"):
                setattr(self, key, value)

    def __repr__(self):
        return f"<Mission: {self.__dict__}>"

