import sys

try:
   x =int(input("x:"))
   y= int(input('y:'))
except ValueError:
    print("Please enter a valid integer.")
    sys.exit(1)

try:
    result = int(x/y)
except ZeroDivisionError:
    print("Division by zero is not allowed.")
    sys.exit(1)

print(f"Result of {x} / {y} is : {result}")