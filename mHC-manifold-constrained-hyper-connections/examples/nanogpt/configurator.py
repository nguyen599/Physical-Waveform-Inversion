import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        with open(arg, "r") as f:
            exec(f.read(), globals())
    else:
        key, value = arg.split("=", 1)
        try:
            value = literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        globals()[key] = value
