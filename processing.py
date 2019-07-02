import multiprocessing as mp
from issou import *

def multiproc(function, nombreproc, *args):
    """
    Exécute la fonction function sur nombreproc processus, avec les arguments *args.
    Les valeurs sont retournées, processus par processus, dans values.
    """
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    manager = mp.Manager()
    values = manager.dict()
    jobs = []
    for i in range(nombreproc):
        p = mp.Process(target=process, args=(function, i, values, *args))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
    print(values)
