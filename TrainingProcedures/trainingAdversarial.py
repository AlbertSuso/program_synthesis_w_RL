import torch
import copy
import torchvision
import numpy as np


from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


def reward_foo(inputs, outputs):
    return torch.sum(inputs*outputs, dim=(1, 2, 3)) / torch.sum(((inputs+outputs)-inputs*outputs), dim=(1, 2, 3))

def lossFunction(actionLogProbabilities, criticValues, entropy, reward, delta=0.6):
    batch_size = actionLogProbabilities[0].shape[0]
    num_steps = len(actionLogProbabilities)
    rewards = torch.zeros((batch_size, num_steps), dtype=torch.float32, device=reward.device)

    loss = -entropy
    for i in range(num_steps):
        factor = 1
        for j in range(i, len(actionLogProbabilities)):
            rewards[:, i] = rewards[:, i] + factor*reward[:, j]
            factor = factor*delta
        loss = loss - actionLogProbabilities[i] * (rewards[:, i] - criticValues[i].view(-1))
    return torch.sum(loss), rewards


def trainAdversarial(environments, dataset, policy, critic, discriminator, num_steps, batch_size, num_epochs, policy_learning_rate, critic_learning_rate, entropy_param, optimizer=Adam, num_experiment=0):
    print("\n\n\n EMPIEZA EL ENTRENAMIENTO DEL AGENTE")
    print("Experiment #{} ---> num_steps={} batch_size={} num_epochs={} policy_learning_rate={} critic_learning_rate={} entropy_param={}".format(num_experiment, num_steps, batch_size, num_epochs, policy_learning_rate, critic_learning_rate, entropy_param))

    debug = False

    gpu1 = torch.device("cuda:0")
    gpu2 = torch.device("cuda:1")
    policy.to(gpu1)
    critic.to(gpu2)
    discriminator.to(gpu2)

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
    policyOptimizer = optimizer(policy.parameters(), lr=policy_learning_rate)
    criticOptimizer = optimizer(critic.parameters(), lr=critic_learning_rate)
    discriminatorOptimizer = optimizer(discriminator.parameters())

    """CONTADORES PARA EL GUARDADO DE DATOS EN TENSORBOARD"""
    if debug:
        counter = 0
        counter_reward = 0

    """AQUI GUARDAREMOS LOS PESOS DE LAS REDES EN SU MEJOR MOMENTO"""
    best_state_dict_policy = None
    best_state_dict_critic = None
    best_reward = -float('inf')

    """EMPEZAMOS EL ENTRENAMIENTO"""
    policy_cummulative_reward = 0.0
    critic_cummulative_loss = 0.0

    for epoch in range(num_epochs):
        print("Starting training epoch #{}".format(epoch))
        for i, data in enumerate(traindataloader):
            # Reseteamos los gradientes de las redes
            policy.zero_grad()
            critic.zero_grad()
            discriminator.zero_grad()

            # Obtenemos las imágenes del batch
            objective_canvas_policy = data[0].to(gpu1)
            objective_canvas_critic = data[0].to(gpu2)

            # Empezamos la generación de la imagen (en 15 steps).
            # Iremos guardando los criticValues y las probabilidades asociadas a las acciones
            log_probabilities = []
            critic_values = []
            discriminator_rewards = torch.zeros((batch_size, num_steps), device=gpu2)
            entropy_sum = 0
            actions = torch.zeros((batch_size, len(environments[0].action_spec)), dtype=torch.long).cuda()
            discriminator_reward = 0
            step_critic = torch.zeros(batch_size, dtype=torch.long, device=gpu2)
            step_policy = torch.zeros(batch_size, dtype=torch.long, device=gpu1)
            for step in range(num_steps):
                # Obtenemos el estado actual del environment
                actual_canvas = torch.cat([torch.from_numpy(environment.observation()["canvas"])
                                          .reshape(1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                                           for environment in environments], dim=0)
                actual_canvas_policy = actual_canvas.to(gpu1)
                actual_canvas_critic = actual_canvas.to(gpu2)

                # Obtenemos la valoración del crítico del estado actual del environment
                last_action_end_position_critic = actions[:, 1].to(gpu2)
                critic_values.append(critic(actual_canvas_critic, objective_canvas_critic, last_action_end_position_critic, step_critic))

                # Generamos la distribucion de probabilidades de la acción, y la usamos para obtener una acción concreta
                last_action_end_position = actions[:, 1].to(gpu1)
                actions, entropy, log_probs = policy(actual_canvas_policy, objective_canvas_policy, last_action_end_position, step_policy)

                entropy_sum = entropy_sum + entropy
                log_probabilities.append(log_probs.to(gpu2))

                # Transformamos las acciones al formato aceptado por el environment y las efectuamos
                for action, environment in zip(actions, environments):
                    for key, value in zip(environmentAction, action):
                        environmentAction[key] = np.array(int(value), dtype=np.int32)
                    environment.step(environmentAction)

                # Obtenemos el reward
                output_images = torch.cat([torch.from_numpy(environment.observation()["canvas"]).reshape(1, environment.num_channels,
                                                                                                         environment.canvas_width,
                                                                                                         environment.canvas_width)
                                           for environment in environments], dim=0)
                output_images.to(gpu2)

                old_reward = discriminator_reward
                discriminator_reward = discriminator(output_images, objective_canvas_critic)
                discriminator_rewards[:, step] = discriminator_reward-old_reward

                step_critic = step_critic+1
                step_policy = step_policy+1

            # Actualizamos el reward
            policy_cummulative_reward += torch.sum(reward_foo(output_images, objective_canvas_critic))

            # Obtenemos la loss de la policy y efectuamos un paso de descenso gradiente
            policy_loss, final_rewards = lossFunction(log_probabilities, critic_values, entropy_param*entropy_sum,
                                                      discriminator_rewards, delta=0.6)

            policy_loss.backward(retain_graph=True)
            policyOptimizer.step()

            # Reseteamos los gradientes almacenados en los parametros de la red critic
            critic.zero_grad()
            discriminator.zero_grad()

            # Obtenemos la loss del critic y efectuamos un paso de descenso gradiente
            critic_values = torch.cat(critic_values, dim=1)
            critic_loss = torch.nn.functional.mse_loss(critic_values, final_rewards)
            critic_cummulative_loss += critic_loss
            critic_loss.backward(retain_graph=True)
            criticOptimizer.step()

            # Reseteamos los gradientes almacenados en los parametros de la red discriminator
            discriminator.zero_grad()
            # Aplicamos el discriminador sobre ejemplos verdaderos
            discriminator_reward_true_examples = discriminator(objective_canvas_critic, objective_canvas_critic)
            discriminator_rewards = torch.cat((discriminator_reward, discriminator_reward_true_examples), dim=0)
            labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)), dim=0)
            discriminator_loss = torch.nn.functional.binary_cross_entropy(discriminator_rewards, labels)
            discriminator_loss.backward()
            discriminatorOptimizer.step()

            # Reseteamos los environments
            for environment in environments:
                environment.reset()

            # Recopilación de datos para análisis de resultados
            if debug and i % int((len(traindataset) // batch_size) // 50) == int((len(traindataset) // batch_size) // 50 - 1):
                entropy = entropy_param * entropy_sum
                writer.add_scalar("Entropy", entropy, counter)
                writer.add_scalar("non-entropy Loss", policy_loss + entropy/batch_size, counter)
                writer.add_scalar("Loss", policy_loss, counter)
                counter += 1

            """GUARDADO PERIODICO DEL REWARD SOBRE TRAINING Y EVAL"""
            if debug and i % int((len(traindataset)//batch_size)//5) == int((len(traindataset)//batch_size)//5 - 1):
                writer.add_scalar("training_mean_reward", policy_cummulative_reward/(batch_size*int((len(traindataset)//batch_size)//5)), counter_reward)
                writer.add_scalar("critic_loss", critic_cummulative_loss, counter_reward)

                policy_cummulative_reward = 0
                critic_cummulative_loss = 0

                #Calculamos el reward medio sobre el evaluation dataset
                with torch.no_grad():
                    for j, validateData in enumerate(evaldataloader):
                        objective_canvas_policy = validateData[0].to(gpu1)
                        actions = torch.zeros((batch_size, len(environments[0]._action_spec)))
                        step_policy = torch.zeros(batch_size, dtype=torch.long, device=gpu1)
                        for step in range(num_steps):
                            actual_canvas = torch.cat([torch.from_numpy(
                                environment.observation()["canvas"]).reshape(1, environment.num_channels,
                                                                             environment.canvas_width,
                                                                             environment.canvas_width).cuda()
                                                           for environment in environments], dim=0)
                            actual_canvas_policy = actual_canvas.to(gpu1)

                            # Ejecutamos la policy
                            last_action_end_position = actions[:, 1].to(gpu1)
                            actions, entropy, log_probs = policy(actual_canvas_policy, objective_canvas_policy, last_action_end_position, step_policy)

                            # Transformamos las acciones al formato aceptado por el environment y las efectuamos
                            for action, environment in zip(actions, environments):
                                for key, value in zip(environmentAction, action):
                                    environmentAction[key] = np.array(int(value), dtype=np.int32)
                                environment.step(environmentAction)

                        output_images = torch.cat([torch.from_numpy(environment.observation()["canvas"]).reshape(
                            1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                                                  for environment in environments], dim=0)
                        output_images.to('gpu1')
                        reward = reward_foo(output_images, objective_canvas_policy)

                        policy_cummulative_reward += torch.sum(reward)

                        for environment in environments:
                            environment.reset()

                        if debug and j == 0:
                            images = output_images[0:16:5]
                            img_grid = torchvision.utils.make_grid(images)
                            writer.add_image('output_images', img_grid)

                            images = objective_canvas_policy[0:16:5]
                            img_grid = torchvision.utils.make_grid(images)
                            writer.add_image('input_images', img_grid)

                    if debug:
                        writer.add_scalar("eval_mean_reward", policy_cummulative_reward/10000, counter_reward)
                        counter_reward += 1

                if policy_cummulative_reward > best_reward:
                    print("Mejora en el reward obtenido! Epoca {}".format(epoch))
                    print("Pasamos de un reward de {} a un reward de {}".format(best_reward, policy_cummulative_reward))

                    best_reward = policy_cummulative_reward
                    best_state_dict_policy = policy.state_dict()
                    best_state_dict_critic = critic.state_dict()

                policy_cummulative_reward = 0



    """PROBAMOS LA RED SOBRE EL CONJUNTO DE VALIDACIÓN"""
    # print("\nEmpezamos la prueba final sobre el conjunto de validación")
    policy_cummulative_reward = 0
    with torch.no_grad():
        for j, validateData in enumerate(evaldataloader):
            objective_canvas_policy = validateData[0].to(gpu1)
            actions = torch.zeros((batch_size, len(environments[0]._action_spec)))
            step_policy = torch.zeros(batch_size, dtype=torch.long, device=gpu1)
            for step in range(num_steps):
                actual_canvas = torch.cat([torch.from_numpy(
                    environment.observation()["canvas"]).reshape(1, environment.num_channels,
                                                                 environment.canvas_width,
                                                                 environment.canvas_width).cuda()
                                           for environment in environments], dim=0)
                actual_canvas_policy = actual_canvas.to(gpu1)

                # Ejecutamos la policy
                last_action_end_position = actions[:, 1].to(gpu1)
                actions, entropy, log_probs = policy(actual_canvas_policy, objective_canvas_policy,
                                                     last_action_end_position, step_policy)

                # Transformamos las acciones al formato aceptado por el environment y las efectuamos
                for action, environment in zip(actions, environments):
                    for key, value in zip(environmentAction, action):
                        environmentAction[key] = np.array(int(value), dtype=np.int32)
                    environment.step(environmentAction)

            output_images = torch.cat([torch.from_numpy(environment.observation()["canvas"]).reshape(
                1, environment.num_channels, environment.canvas_width, environment.canvas_width)
                for environment in environments], dim=0)
            output_images.to('gpu1')
            reward = reward_foo(output_images, objective_canvas_policy)

            policy_cummulative_reward += torch.sum(reward)

            for environment in environments:
                environment.reset()

        last_reward = policy_cummulative_reward / ((len(evaldataset) // batch_size) * batch_size)

    if last_reward > best_reward:
        best_reward = last_reward
        best_state_dict_policy = policy.state_dict()
        best_state_dict_critic = critic.state_dict()

    """GENERAMOS Y GUARDAMOS LAS GRAFICAS PARA EL ANALISIS DEL ENTRENAMIENTO. GUARDAMOS LOS BEST STATE DICTS"""

    torch.save(best_state_dict_critic,
               "state_dicts/experiment"+str(num_experiment)+"_critic")
    torch.save(best_state_dict_policy,
               "state_dicts/experiment"+str(num_experiment)+"_policy")

    if debug:
        writer.close()

    return best_reward