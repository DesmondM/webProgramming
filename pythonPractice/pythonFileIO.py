fruit = input("Enter fruit name: ")

with open("fruits.txt", 'a') as file:
    file.write(f"{fruit}\n")