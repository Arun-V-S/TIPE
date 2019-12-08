Modèle 1 
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
