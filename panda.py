from direct.showbase.ShowBase import ShowBase
SPAWNED = False
class Window(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.loadModels()

    def loadModels(self):
        self.ttc = loader.loadModel("Panda\models\enrironment.egg")
        self.ttc.reparentTo(render)

game = Window()
game.run()
