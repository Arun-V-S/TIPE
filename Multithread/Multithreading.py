import time
import random
import sys
import threading
import math

def CPUBurner(n):
    for i in range(n):
        for j in range(n):
            a = math.sqrt(j) ** i




class Mythread(fonction, *args, threading.Thread):
    def run(self):
        print("{} started!".format(self.getName()))
        self.fonction(args)
        print("{} finished!".format(self.getName()))

def main(nbThreads, fonction, *args):
    for x in range(nbThreads):
        mythread = Mythread(fonction, *args, name = "Thread-{}".format(x + 1))
        mythread.start()
        time.sleep(.5)

def Issou(a, b):
    return a + b
