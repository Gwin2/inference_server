import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchensemble import VotingClassifier


class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear3 = nn.Linear(32, 3)

    def forward(self, data):
        data = data.view(data.size(0), -1)
        output = F.relu(self.linear1(data))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train = datasets.VisionDataset('../Dataset', train=True, download=True, transform=transform)
test = datasets.VisionDataset('../Dataset', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

model = VotingClassifier(estimator=Ensemble, n_estimators=10, cuda=True)

criterion = nn.CrossEntropyLoss()
model.set_criterion(criterion)

model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

model.fit(train_loader, epochs=50, test_loader=test_loader)
