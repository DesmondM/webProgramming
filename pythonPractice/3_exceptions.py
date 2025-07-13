def main():
    x = get_int()
    print(f'x is {x}')

def get_int():
    while True:
        try:
            return  int(input('Enter a number x \n'))
        except ValueError:
            pass

main()