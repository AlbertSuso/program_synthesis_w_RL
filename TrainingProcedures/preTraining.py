import torch
import torch.nn as nn


from torch.utils.data import Subset, DataLoader
from torch.optim import Adam


def preTrainFeatureExtractor(feature_extractor, dataset, batch_size, num_epochs, optimizer=Adam, cuda=True):
    # print("Començem el pre-entrenament del feature extractor\n")
    feature_extractor.train()
    torch.manual_seed(0)
    idxs = torch.randperm(len(dataset))
    evaldataset = Subset(dataset, idxs[:int(len(dataset) / 6)])
    traindataset = Subset(dataset, idxs[int(len(dataset) / 6):])
    # Generamos los dataloaders
    traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    evaldataloader = DataLoader(evaldataset, batch_size=batch_size, pin_memory=True, drop_last=False)

    optimizer = optimizer(feature_extractor.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    training_losses = []
    eval_losses = []

    running_loss = 0
    total = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(traindataloader):
            data, label = data
            if cuda:
                data = data.cuda()
                label = label.cuda()
            output = feature_extractor(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            feature_extractor.zero_grad()

            running_loss += loss
            total += label.size(0)

            """
            if i % (10000//batch_size) == 10000//batch_size-1:
                print("{}.{} La loss mitjana sobre les últimes {} dades és {}".format(epoch, i//(10000//batch_size), total, running_loss/total))
                training_losses.append(running_loss/total)
                running_loss = 0
                total = 0

                with torch.no_grad():
                    for data, label in evaldataloader:
                        if cuda:
                            data = data.cuda()
                            label = label.cuda()
                        output = feature_extractor(data)
                        running_loss += criterion(output, label)
                        total += label.size(0)
                    print("{}.{} La loss mitjana sobre el conjunt de validació és {}".format(epoch,
                                                                                          i // (10000 // batch_size),
                                                                                          running_loss/total))
                    eval_losses.append(running_loss/total)
                    running_loss = 0
                    total = 0


    # print("\nEl pre-entrenament del feature extractor ha finalitzat\n")


    plot1, = plt.plot(training_losses, 'r', label="train_loss")
    plot2, = plt.plot(eval_losses, 'b', label="eval_loss")
    plt.legend(handles=[plot1, plot2])
    plt.show()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in evaldataloader:
            if cuda:
               inputs, labels = inputs.cuda(), labels.cuda()

            predictions = feature_extractor(inputs)

            _, predictions = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    print("Accuracy of the network over the eval data is: ", (100 * correct / total))
    """

    feature_extractor.eval()  # Elimina la capa de clasificación