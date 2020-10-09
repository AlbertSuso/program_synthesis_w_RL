import torch
import copy
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image
from torch.optim import Adam


def getAction(distributions):
    return [random.choices(list(range(len(distribution))), weights=distribution)[0] for distribution in distributions]


def lossFunction(actionProbabilities, criticValues, entropy_param, entropy_sum, badReward):
    log_probabilities = [torch.log(prob) for prob in actionProbabilities]
    loss = -entropy_param*entropy_sum
    print("La entropy_sum part es", torch.sum(loss)/len(loss))
    for i in range(len(actionProbabilities)):
        loss = loss - log_probabilities[i] * (badReward - criticValues[i].view(-1))
    print("La loss total es", torch.sum(loss)/len(loss))
    return torch.sum(loss)/len(loss)


def preTrainFeatureExtractor(feature_extractor, dataset, batch_size, num_epochs, optimizer=Adam, cuda=True):
    print("Començem el pre-entrenament del feature extractor\n")
    feature_extractor.train()
    torch.manual_seed(0)
    idxs = torch.randperm(len(dataset))
    evaldataset = Subset(dataset, idxs[:int(len(dataset) / 6)])
    traindataset = Subset(dataset, idxs[int(len(dataset) / 6):])
    # Generamos los dataloaders
    traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
    evaldataloader = DataLoader(evaldataset, batch_size=batch_size, pin_memory=True, drop_last=False)

    optimizer = optimizer(feature_extractor.parameters(), lr=0.0001)
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

    print("\nEl pre-entrenament del feature extractor ha finalitzat\n")

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
    feature_extractor.eval()


def train(environments, dataset, feature_extractor, policy, critic, discriminator, num_steps, batch_size, num_epochs, optimizer=Adam, entropy_param=0.01):
    print("\n\n\n EMPIEZA EL ENTRENAMIENTO DEL AGENTE\n")

    # Obtenemos el formato deseado para las acciones
    environmentAction = copy.copy(environments[0].action_spec)


    """PREPARADO DE LOS DATOS"""
    # Establecemos una seed y desordenamos los elementos del dataset. Lo dividimos en train y eval.
    torch.manual_seed(0)
    idxs = torch.randperm(len(dataset))
    evaldataset = Subset(dataset, idxs[:int(len(dataset)/6)])
    traindataset = Subset(dataset, idxs[int(len(dataset)/6):])
    # Generamos los dataloaders
    traindataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    evaldataloader = DataLoader(evaldataset, batch_size=batch_size, pin_memory=True, drop_last=True)

    """INSTANCIAMOS LOS OPTIMIZADORES Y LA FUNCIÓN DE PERDIDA (MSELoss)"""
    policyOptimizer = optimizer(list(feature_extractor.parameters())+list(policy.parameters()))
    criticOptimizer = optimizer(critic.parameters())

    """EMPEZAMOS EL ENTRENAMIENTO"""
    policy_cummulative_loss = 0.0

    for epoch in range(num_epochs):
        print("Starting training epoch #{}".format(epoch))
        for i, data in enumerate(traindataloader):
            # Reseteamos los gradientes de las redes
            feature_extractor.zero_grad()
            policy.zero_grad()
            critic.zero_grad()

            # Obtenemos una imagen y sus features
            inputImages = data[0].cuda()
            #inputFeatures = feature_extractor(torch.cat([inputImages]*3, dim=1))
            inputFeatures = feature_extractor(inputImages)

            # Empezamos la generación de la imágen (en 15 steps). Iremos guardando los criticValues y las probabilidades asociadas a las acciones
            probabilities = []
            criticValues = []
            for step in range(num_steps):
                # Obtenemos el estado actual del environment
                environmentStates = torch.cat([torch.from_numpy(environment.observation()["canvas"]).reshape(1, environment.num_channels, environment.canvas_width, environment.canvas_width).cuda()
                                     for environment in environments], dim=0)

                # Obtenemos los features del estado actual del environment
                #canvasFeatures = feature_extractor(torch.cat([environmentStates]*3, dim=1))
                canvasFeatures = feature_extractor(environmentStates)

                # Obtenemos la valoración del crítico del estado actual del environment
                criticValues.append(critic(canvasFeatures, inputFeatures, (step+1)/num_steps))

                # Generamos la distribucion de probabilidades de la acción, y la usamos para obtener una acción concreta
                distributions = policy(canvasFeatures, inputFeatures)

                """AQUI EMPIEZA COMPROVACIÓN"""
                actions_with_max_probs = []
                if i == 0 or i == 311 or i == 1247:
                    for k in [0, 8, 16, 24]:
                        actions_with_max_probs.append(
                            [(torch.argmax(distribution[k]), distribution[k][torch.argmax(distribution[k])])
                             for distribution in distributions])
                """AQUI TERMINA COMPROVACIÓN"""
                entropies = torch.stack([torch.distributions.Categorical(distribution).entropy()
                                         for distribution in distributions], dim=1)
                entropy_sum = torch.sum(entropies, dim=1) #Fusionar el sum i el stack
                actions = torch.stack([torch.distributions.Categorical(distribution).sample()
                                       for distribution in distributions], dim=1)

                """AQUI SIGUIE COMPROVACIÓN"""
                if i == 0:
                        print("Epoch {} comprovation 0 step {}".format(epoch, step))
                if i == 311:
                        print("Epoch {} comprovation 1 step {}".format(epoch, step))
                elif i == 1247:
                    print("Epoch {} comprovation 2 step {}".format(epoch, step))
                if i == 0 or i == 311 or i == 1247:
                    for action_probs, action_done in zip(actions_with_max_probs, actions):
                        print("Les accions amb probabilitat máxima, junt amb la seva probabilitat son:\n", action_probs)
                        print("L'acció realitzada és:", action_done, "\n\n")
                """AQUI TERMINA"""

                # Calculamos la probabilidad de la acción obtenida
                actionProbabilities = torch.ones(batch_size, dtype=torch.float32).cuda()
                for sample in range(batch_size):
                    for j, act in enumerate(actions[sample]):
                        actionProbabilities[sample] *= distributions[j][sample, act]
                probabilities.append(actionProbabilities)

                # Transformamos las acciones al formato aceptado por el environment y las efectuamos
                for action, environment in zip(actions, environments):
                    for key, value in zip(environmentAction, action):
                        environmentAction[key] = np.array(int(value), dtype=np.int32)
                    environment.step(environmentAction)

            # Obtenemos la loss de la policy y efectuamos un paso de descenso gradiente
            outputImages = torch.cat([torch.from_numpy(environment.observation()["canvas"]).cuda().reshape(1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                                      for environment in environments], dim=0)
            badRewards = discriminator(outputImages, inputImages)

            policy_cummulative_loss += torch.sum(badRewards)
            policyLoss = lossFunction(probabilities, criticValues, entropy_param, entropy_sum, badRewards)
            policyLoss.backward(retain_graph=True)

            # Obtenemos la loss del critic y efectuamos un paso de descenso gradiente
            criticLoss = torch.nn.functional.mse_loss(torch.stack(criticValues, dim=0).reshape(num_steps, -1), torch.stack([badRewards]*num_steps, dim=0))
            criticLoss.backward()

            policyOptimizer.step()
            criticOptimizer.step()

            # Reseteamos los environments
            for environment in environments:
                environment.reset()

            """PRINTEO PERIODICO DE LA MSEloss SOBRE TRAINING Y EVAL"""
            if i % 312 == 311:
                print("La loss media de la policy sobre las ultimas 10000 imagenes es {}".format(
                    policy_cummulative_loss / 10048))
                policy_cummulative_loss = 0
                with torch.no_grad():
                    for validateData in evaldataloader:
                        inputImages = validateData[0].cuda()
                        inputFeatures = feature_extractor(inputImages)
                        for step in range(num_steps):
                            environmentStates = torch.cat([torch.from_numpy(
                                environment.observation()["canvas"]).reshape(1, environment.num_channels,
                                                                             environment.canvas_width,
                                                                             environment.canvas_width).cuda()
                                                           for environment in environments], dim=0)

                            canvasFeatures = feature_extractor(environmentStates)
                            distributions = policy(canvasFeatures, inputFeatures)

                            actions = torch.stack([torch.distributions.Categorical(distribution).sample() for distribution in distributions], dim=1)

                            for action, environment in zip(actions, environments):
                                for key, value in zip(environmentAction, action):
                                    environmentAction[key] = np.array(int(value), dtype=np.int32)
                                environment.step(environmentAction)

                        outputImages = torch.cat([torch.from_numpy(environment.observation()["canvas"]).cuda().reshape(
                            1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                                                  for environment in environments], dim=0)
                        badRewards = discriminator(outputImages, inputImages)

                        policy_cummulative_loss += torch.sum(badRewards)
                        for environment in environments:
                            environment.reset()

                    print("La loss media de la policy sobre el conjunto de validación es {}".format(
                        policy_cummulative_loss / 10000))
                    policy_cummulative_loss = 0

                if i == 311 or i == 1247:
                    for k in [0, 8, 16, 24]:
                        save_image(outputImages[k], "Imatges/epoch{}_{}.png".format(epoch, int(k/8)))

        torch.save(feature_extractor.state_dict(), "StateDicts/epoch{}_featureExtractor".format(epoch))
        torch.save(critic.state_dict(), "StateDicts/epoch{}_critic".format(epoch))
        torch.save(policy.state_dict(), "StateDicts/epoch{}_policy".format(epoch))


    """PROBAMOS LA RED SOBRE EL CONJUNTO DE VALIDACIÓN"""
    print("\nEmpezamos la prueba final sobre el conjunto de validación")
    with torch.no_grad():
        for validateData in evaldataloader:
            inputImages = validateData[0].cuda()
            inputFeatures = feature_extractor(inputImages)
            for step in range(num_steps):
                environmentStates = torch.cat([torch.from_numpy(
                    environment.observation()["canvas"]).reshape(1, environment.num_channels,
                                                                 environment.canvas_width,
                                                                 environment.canvas_width).cuda()
                                               for environment in environments], dim=0)

                canvasFeatures = feature_extractor(environmentStates)
                distributions = policy(canvasFeatures, inputFeatures)
                actions = torch.stack(
                    [torch.distributions.Categorical(distribution).sample() for distribution in distributions], dim=1)

                for action, environment in zip(actions, environments):
                    for key, value in zip(environmentAction, action):
                        environmentAction[key] = np.array(int(value), dtype=np.int32)
                    environment.step(environmentAction)

            outputImages = torch.cat([torch.from_numpy(environment.observation()["canvas"]).cuda().reshape(
                1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                for environment in environments], dim=0)
            badRewards = discriminator(outputImages, inputImages)

            policy_cummulative_loss += torch.sum(badRewards)
            for environment in environments:
                environment.reset()

        print("La loss media de la policy en la prueba definitiva sobre el conjunto de validación es {}".format(
            policy_cummulative_loss / 10000))
