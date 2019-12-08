#Modèle 1 (2 025 280)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.epochs = 0

        self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.review(x)

        x = self.classifier(x)
        return x

    def review(self, x):
        return x.view(-1, 2048)
    
#Modèle 2 (478 544)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.epochs = 0

        self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 4, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 64, 4, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 64, 4, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, 4, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 64, 4, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 64, 4, 2, 2),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 64, 4, 2, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.review(x)

        x = self.classifier(x)
        return x

    def review(self, x):
        return x.view(-1, 1024)
    
    #Modèle 3 (253 568)
    class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.epochs = 0

        self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(4, 4),
        nn.Conv2d(16, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(4, 4),

        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(4, 4),

        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.review(x)

        x = self.classifier(x)
        return x

    def review(self, x):
        return x.view(-1, 1024)
    
#Modèle 4 (34 432)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.epochs = 0

        self.conv = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(4, 4),
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(4, 4),

        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(4, 4),

        nn.Conv2d(32, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(4, 4),
        )

        self.classifier = nn.Sequential(
        nn.Linear(128, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.review(x)

        x = self.classifier(x)
        return x

    def review(self, x):
        return x.view(-1, 128)

    #Modèle 5 (2 706)
    class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.epochs = 0

        self.conv = nn.Sequential(
        nn.Conv2d(3, 8, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(4, 4),
        nn.Conv2d(8, 8, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(4, 4),

        nn.Conv2d(8, 8, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 8, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(4, 4),

        nn.Conv2d(8, 8, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.MaxPool2d(4, 4),
        )

        self.classifier = nn.Sequential(
        nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.conv(x)

        x = self.review(x)

        x = self.classifier(x)
        return x

    def review(self, x):
        return x.view(-1, 32)
