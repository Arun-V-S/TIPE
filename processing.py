import multiprocessing as mp
from issou import *

def calc_square(i, number, dict):
    #print('Square:' , number * number)
    result = number * number
    #print(result)
    dict[i] = result

def burn(i):
    a = 0
    for j in range(i ** 2):
        a += 0


def calc_quad(number):
    print('Quad:' , number * number * number * number)



def multiproc(function, nombreproc, *args):
    """Exécute la fonction function sur i processus, avec les arguments *args."""
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
    return values






    #mp.set_start_method('spawn')
    #q = mp.Queue()
    #p = mp.Process(target=calc_square, args=(0, number, return_dict))
    #p.start()
    #print(q.get())
    #p.join()
    #print(return_dict.values())
    #return_dict
    #q.join()

    # Wont print because processes run using their own memory location
#print(result)
