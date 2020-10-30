import torch
import copy
import random
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


def lossFunction(actionLogProbabilities, criticValues, entropy, reward):
    loss = -entropy
    for i in range(len(actionLogProbabilities)):
        loss = loss - actionLogProbabilities[i] * (reward - criticValues[i].view(-1))
    return torch.sum(loss)/len(loss)


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

    feature_extractor.eval() #Elimina la capa de clasificación


def train(environments, dataset, feature_extractor_policy, feature_extractor_critic, policy, critic, discriminator, num_steps, batch_size, num_epochs, policy_learning_rate, critic_learning_rate, entropy_param, optimizer=Adam, num_experiment=0):
    print("\n\n\n EMPIEZA EL ENTRENAMIENTO DEL AGENTE")
    print("Experiment #{} ---> num_steps={} batch_size={} num_epochs={} policy_learning_rate={} critic_learning_rate={} entropy_param={}".format(num_experiment, num_steps, batch_size, num_epochs, policy_learning_rate, critic_learning_rate, entropy_param))

    debug = False

    if debug:
        # Tensorboard writter
        writer = SummaryWriter("graphics/experiment"+str(num_experiment))

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
    policyOptimizer = optimizer(list(feature_extractor_policy.parameters())+list(policy.parameters()), lr=policy_learning_rate)
    criticOptimizer = optimizer(list(feature_extractor_critic.parameters())+list(critic.parameters()), lr=critic_learning_rate)

    """CONTADORES PARA EL GUARDADO DE DATOS EN TENSORBOARD"""
    if debug:
        counter = 0
        counter_reward = 0

    """AQUI GUARDAREMOS LOS PESOS DE LAS REDES EN SU MEJOR MOMENTO"""
    best_state_dict_policy = None
    best_state_dict_critic = None
    best_state_dict_feature_extractor_policy = None
    best_state_dict_feature_extractor_critic = None
    best_reward = -float('inf')

    """EMPEZAMOS EL ENTRENAMIENTO"""
    policy_cummulative_reward = 0.0
    critic_cummulative_loss = 0.0

    for epoch in range(num_epochs):
        print("Starting training epoch #{}".format(epoch))
        for i, data in enumerate(traindataloader):
            # Reseteamos los gradientes de las redes
            feature_extractor_policy.zero_grad()
            feature_extractor_critic.zero_grad()
            policy.zero_grad()
            critic.zero_grad()

            # Obtenemos una imagen y sus features
            inputImages = data[0].cuda()
            inputFeatures_policy = feature_extractor_policy(inputImages)
            inputFeatures_critic = feature_extractor_critic(inputImages)

            # Empezamos la generación de la imagen (en 15 steps). Iremos guardando los criticValues y las probabilidades asociadas a las acciones
            probabilities = []
            criticValues = []
            entropy_sum = 0
            for step in range(num_steps):
                # Obtenemos el estado actual del environment
                environmentStates = torch.cat([torch.from_numpy(environment.observation()["canvas"]).reshape(1, environment.num_channels, environment.canvas_width, environment.canvas_width).cuda()
                                     for environment in environments], dim=0)

                # Obtenemos los features del estado actual del environment
                canvasFeatures_policy = feature_extractor_policy(environmentStates)
                canvasFeatures_critic = feature_extractor_critic(environmentStates)

                # Obtenemos la valoración del crítico del estado actual del environment
                criticValues.append(critic(canvasFeatures_critic, inputFeatures_critic, (step+1)/num_steps))

                # Generamos la distribucion de probabilidades de la acción, y la usamos para obtener una acción concreta
                distributions = policy(canvasFeatures_policy, inputFeatures_policy)

                entropies = torch.stack([torch.distributions.Categorical(distribution).entropy()
                                         for distribution in distributions], dim=1)

                entropy_sum += torch.sum(entropies, dim=1)
                actions = torch.stack([torch.distributions.Categorical(distribution).sample()
                                       for distribution in distributions], dim=1)

                # Calculamos la log-probabilidad de la acción obtenida
                actionLogProbabilities = torch.zeros(batch_size, dtype=torch.float32).cuda()
                for sample in range(batch_size):
                    for j, act in enumerate(actions[sample]):
                        actionLogProbabilities[sample] += torch.log(distributions[j][sample, act])
                probabilities.append(actionLogProbabilities)

                # Transformamos las acciones al formato aceptado por el environment y las efectuamos
                for action, environment in zip(actions, environments):
                    for key, value in zip(environmentAction, action):
                        environmentAction[key] = np.array(int(value), dtype=np.int32)
                    environment.step(environmentAction)

            # Obtenemos el reward
            outputImages = torch.cat([torch.from_numpy(environment.observation()["canvas"]).cuda().reshape(1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                                      for environment in environments], dim=0)
            reward = discriminator(outputImages, inputImages)
            policy_cummulative_reward += torch.sum(reward)

            # Obtenemos la loss de la policy y efectuamos un paso de descenso gradiente
            policyLoss = lossFunction(probabilities, criticValues, entropy_param*entropy_sum, reward)

            policyLoss.backward(retain_graph=True)
            policyOptimizer.step()

            # Reseteamos los gradientes almacenados en los parametros de las redes critic y feature_Extractor_critic
            critic.zero_grad()
            feature_extractor_critic.zero_grad()

            # Obtenemos la loss del critic y efectuamos un paso de descenso gradiente
            criticLoss = torch.nn.functional.mse_loss(torch.stack(criticValues, dim=0).reshape(num_steps, -1), torch.stack([reward]*num_steps, dim=0))
            critic_cummulative_loss += torch.sum(criticLoss)
            criticLoss.backward()
            criticOptimizer.step()

            # Reseteamos los environments
            for environment in environments:
                environment.reset()

            # Recopilación de datos para análisis de resultados
            if debug and i % int((len(traindataset) // batch_size) // 50) == int((len(traindataset) // batch_size) // 50 - 1):
                entropy = entropy_param * entropy_sum
                writer.add_scalar("Entropy/train", entropy, counter)
                writer.add_scalar("non-entropy Loss/train", policyLoss + entropy, counter)
                counter += 1

            if debug and i % int(len(traindataset) // batch_size) == int(len(traindataset) // batch_size)-1:
                images = outputImages[0:16:5]
                img_grid = torchvision.utils.make_grid(images)
                writer.add_image('training_images', img_grid)

            """GUARDADO PERIODICO DEL REWARD SOBRE TRAINING Y EVAL"""
            if i % int((len(traindataset)//batch_size)//5) == int((len(traindataset)//batch_size)//5 - 1):
                if debug:
                    writer.add_scalar("training_mean_reward", policy_cummulative_reward, counter_reward)
                    writer.add_scalar("critic_loss",
                                      critic_cummulative_loss / (batch_size * ((len(traindataset) // batch_size) // 5)),
                                      counter_reward)

                policy_cummulative_reward = 0
                critic_cummulative_loss = 0

                #Calculamos el reward medio sobre el evaluation dataset
                with torch.no_grad():
                    for validateData in evaldataloader:
                        inputImages = validateData[0].cuda()
                        inputFeatures = feature_extractor_policy(inputImages)
                        for step in range(num_steps):
                            environmentStates = torch.cat([torch.from_numpy(
                                environment.observation()["canvas"]).reshape(1, environment.num_channels,
                                                                             environment.canvas_width,
                                                                             environment.canvas_width).cuda()
                                                           for environment in environments], dim=0)

                            canvasFeatures = feature_extractor_policy(environmentStates)
                            distributions = policy(canvasFeatures, inputFeatures)

                            actions = torch.stack([torch.distributions.Categorical(distribution).sample() for distribution in distributions], dim=1)

                            for action, environment in zip(actions, environments):
                                for key, value in zip(environmentAction, action):
                                    environmentAction[key] = np.array(int(value), dtype=np.int32)
                                environment.step(environmentAction)

                        outputImages = torch.cat([torch.from_numpy(environment.observation()["canvas"]).cuda().reshape(
                            1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                                                  for environment in environments], dim=0)
                        reward = discriminator(outputImages, inputImages)

                        policy_cummulative_reward += torch.sum(reward)

                        for environment in environments:
                            environment.reset()

                    if debug:
                        writer.add_scalar("eval_mean_reward", policy_cummulative_reward, counter_reward)
                        counter_reward += 1

                if policy_cummulative_reward > best_reward:
                    print("Mejora en el reward obtenido! Epoca {}".format(epoch))
                    print("Pasamos de un reward de {} a un reward de {}".format(best_reward, policy_cummulative_reward))

                    best_reward = policy_cummulative_reward
                    best_state_dict_policy = policy.state_dict()
                    best_state_dict_critic = critic.state_dict()
                    best_state_dict_feature_extractor_policy = feature_extractor_policy.state_dict()
                    best_state_dict_feature_extractor_critic = feature_extractor_critic.state_dict()

                policy_cummulative_reward = 0



    """PROBAMOS LA RED SOBRE EL CONJUNTO DE VALIDACIÓN"""
    # print("\nEmpezamos la prueba final sobre el conjunto de validación")
    policy_cummulative_reward = 0
    with torch.no_grad():
        for validateData in evaldataloader:
            inputImages = validateData[0].cuda()
            inputFeatures = feature_extractor_policy(inputImages)
            for step in range(num_steps):
                environmentStates = torch.cat([torch.from_numpy(
                    environment.observation()["canvas"]).reshape(1, environment.num_channels,
                                                                 environment.canvas_width,
                                                                 environment.canvas_width).cuda()
                                               for environment in environments], dim=0)

                canvasFeatures = feature_extractor_policy(environmentStates)
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
            reward = discriminator(outputImages, inputImages)

            policy_cummulative_reward += torch.sum(reward)

            for environment in environments:
                environment.reset()

        last_reward = policy_cummulative_reward / ((len(evaldataset) // batch_size) * batch_size)

    if last_reward > best_reward:
        best_reward = last_reward
        best_state_dict_policy = policy.state_dict()
        best_state_dict_critic = critic.state_dict()
        best_state_dict_feature_extractor_policy = feature_extractor_policy.state_dict()
        best_state_dict_feature_extractor_critic = feature_extractor_critic.state_dict()

    """GENERAMOS Y GUARDAMOS LAS GRÁFICAS PARA EL ANÁLISIS DEL ENTRENAMIENTO. GUARDAMOS LOS BEST STATE DICTS"""

    torch.save(best_state_dict_feature_extractor_critic,
               "state_dicts/experiment"+str(num_experiment)+"_critic_feature_extractor")
    torch.save(best_state_dict_feature_extractor_policy,
               "state_dicts/experiment"+str(num_experiment)+"_policy_feature_extractor")
    torch.save(best_state_dict_critic,
               "state_dicts/experiment"+str(num_experiment)+"_critic")
    torch.save(best_state_dict_policy,
               "state_dicts/experiment"+str(num_experiment)+"_policy")

    writer.close()

    return best_reward
