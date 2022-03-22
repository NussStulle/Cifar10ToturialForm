
# Loading

# Importierung
import torch
import torchvision
import time
import torchvision.transforms as transforms

# Vordifinierungen

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 5

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Import für Bilderanzeige

import matplotlib.pyplot as plt
import numpy as np

#  Funktion zum Zeigen der Bilder

def imshow(img):
    img = img / 2 + 0.5     # Entnormalisierung
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Zufällige Auswahl von Beispielsbilder
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Anzeige der Bilder
imshow(torchvision.utils.make_grid(images))
# Bennenung der Bilder
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



#CNN-Definition

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub() #Quantisierung
        self.conv1 = nn.Conv2d(3, 64, 5, padding='same') # 1. Convolution mit je ein Gangangskanal je
                                                        # RGB- Wert, also 3 Ein- und 64 Ausgangskanälen
        self.conv2 = nn.Conv2d(64, 128, 5, padding='same') # 2. Convolution mit 64 Ein- und 128 Ausgangskanälen
        self.conv3 = nn.Conv2d(128, 256, 5) # 3. Convolution mit 128 Ein- und 256 Ausgangskanälen
        self.pool = nn.MaxPool2d(2, 2) # Maxpool-Defintion mit 2x2 Kernel
        self.conv4 = nn.Conv2d(256, 512, 5) # 4. Convolution mit 256 Ein- und 512 Ausgangskanäle
        self.fc1 = nn.Linear(512 * 5 * 5, 2048) # 1. FullyConnected-Layer mit Kanalgröße * Matrixgröße
                                                # als Eingangskanalgröße und 2048 Ausgangskanäle
        self.fc2 = nn.Linear(2048, 512)  # 2. FullyConntected-Layer mit 2048 Ein- und 512 Ausgangskaäle
        self.fc3 = nn.Linear(512, 128)   # 3. FullycConntected-Layer 512 Ein- und 128 Ausgangskanäle
        self.fc4 = nn.Linear(128, 10)   # 4. FullyConnected
        self.conv_dropout = nn.Dropout2d() # Dropout, um  Zufällig Kanäle zu Löschen für Overcommiting zu Verhindern
        self.dequant = torch.quantization.DeQuantStub() # Dequantisierung


    def forward(self, x):
        x = self.quant(x)  #Quanisierung
        x = self.conv1(x) # Aufruf der 1. Convolution
        x = F.relu(x) #ReLu der Werte
        x = self.conv2(x) # Aufruf der 2. Convolution
        x = F.relu(x) #ReLu der Werte
        x = self.pool(F.relu(self.conv3(x))) # Aufruf der 3. Convolution mit ReLu und Maxpool
        x = self.pool(F.relu(self.conv4(x))) # Aufruf der 4. Convolution mit Relu und Maxpool
        x = self.conv_dropout(x) # Aufruf des  Dropouts
        x = torch.flatten(x, 1) # Vektorisierung der Daten, bis auf den Batch
        x = F.relu(self.fc1(x)) # Aufruf des 1. FullyConnected-Layer mit ReLu
        x = F.relu(self.fc2(x)) # Aufruf des 2. FullyConnected-Layer mit ReLu
        x = F.relu(self.fc3(x)) # Aufruf des 3. FullyConnected-Layer mit ReLu
        x = self.fc4(x) # Aufruf des 4. FullyConnected-Layer
        x = self.dequant(x) #Dequantisierung
        return x

# Quantisierung

model_fp32 = Net()
model_fp32.eval()
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_int8 = torch.quantization.convert(model_fp32)
res = model_int8

net = Net()

# Definitionsschritt für Loss-Berechnung und der Optimierung mit der Lernrate lr

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


# Abfrage, ob das Training über die CPU oder über die GPU läuft, mit Ausgabe des Gerätes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)


# Training beginnt hier per Forwardpopagation

for epoch in range(100):  # Loop über die Anzahl der hier Festgelegte Epochen
    start_time = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Übername der Inputs als Liste der Form [inputs, labels]
        inputs, labels = data
        net.to(device)
        inputs, labels = data[0].to(device), data[1].to(device)

        # Nullen des Parameter-Gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Ausgabe der Statistiken für die Jeweilige Epoche
        running_loss += loss.item()
        if i % 2000 == 1999:    # Ausgabe nach je 2000 Mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
            running_loss = 0.0
    print("--- %s Sekunden  ---" % (time.time() - start_time))

print('Ende des Training')

# Speichern der Ergebnisse

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)



#Evaluierung

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Vorhergesagt: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        # Berechnung der Outputs, in dem Bilder durch das Netzwerk laufen
        outputs = net(images)
        # Die Klasse mit dem höchsten Wert wird als Vorhersage bestimmt
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Genauigkeit des CNNs bei 10000 Testbildern: {100 * correct // total} %')

# Vorbereitung zur Vorhersage für alle Klassen
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # Sammeln der richtigen Vorhersagen je Klasse
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# Ausgabe der Genauigkeit für jede Klasse
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Genauigkeit der für die Klasse: {classname:5s} ist {accuracy:.1f} %')


