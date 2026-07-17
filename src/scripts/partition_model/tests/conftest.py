import os
import sys

# Torna os módulos do pipeline (regret, data, features, partition_defs)
# importáveis sem instalar o pacote.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
