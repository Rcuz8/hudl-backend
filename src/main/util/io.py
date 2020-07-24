from colorama import Fore # , Back, Style

def err(*args):
    s = ''
    for arg in args:
        s += str(arg) + ' '
    print(Fore.RED + 'ERR: ' + s)
def ok(*args):
    s = ''
    for arg in args:
        s += str(arg) + ' '
    print(Fore.LIGHTGREEN_EX + 'OK: ' + s)
def info(*args):
    s = ''
    for arg in args:
        s += str(arg) + ' '
    print(Fore.BLUE + 'INFO: ' + s)
def warn(*args):
    s = ''
    for arg in args:
        s += str(arg) + ' '
    print(Fore.YELLOW + 'WARNING: ' + s)
