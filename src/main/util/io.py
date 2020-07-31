from colorama import Fore # , Back, Style

__log_lvl = 2

def set_log_level(to):
    if to < 0 or to > 2:
        raise Exception('Invalid log level')
    global __log_lvl
    __log_lvl = to
    white("\nAdjusted Log Level.\n\n")

def get_log_level():
    global __log_lvl
    return __log_lvl


def white(*args):
    __fore_print(30, args)

def dkblue(*args):
    __fore_print(36, args)
def lgtblue(*args):
    __fore_print(96, args)

def purple(*args):
    __fore_print(35, args)

def pink(*args):
    __fore_print(95, args)

def err(*args):
    s = ''
    for arg in args:
        s += str(arg) + ' '
    print(Fore.RED + 'ERR: ' + s)

ok = lgtblue
def info(*args):
    global __log_lvl
    if __log_lvl > 1:
        s = ''
        for arg in args:
            s += str(arg) + ' '
        print(Fore.BLUE + 'INFO: ' + s)
def warn(*args):
    global __log_lvl
    if __log_lvl > 0:
        s = ''
        for arg in args:
            s += str(arg) + ' '
        print(Fore.YELLOW + 'WARNING: ' + s)

def __fore_print(code, args):
    s = ''
    for arg in args:
        s += str(arg) + ' '
    print('\33[{}m'.format(code) + s)


