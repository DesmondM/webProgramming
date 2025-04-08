def announce(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} completed.")
        return result
    return wrapper

@announce
def greet(name):
    print(f"Hello, {name}!")

name = "Desmond"
greet(name)
