def get_fields(obj):
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        raise ValueError("The provided object has no fields (no __dict__ attribute).")


# Example usage
class MyClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def f(self):
        pass


obj = MyClass(10, 20)
print(get_fields(obj))
