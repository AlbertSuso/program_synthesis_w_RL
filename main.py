import argparse
import copy
import torch
import torchvision.transforms as transforms
import nets.feature_extractors as featureExtractors
import nets.critics as critics
import nets.policies as policies


from torchvision.datasets import MNIST
from LibMyPaint.LibMyPaint import Environment
from torch.optim import Adam
from TrainingProcedures.training import trainSimple
from TrainingProcedures.trainingRecurrent import trainRecurrent
from TrainingProcedures.preTraining import preTrainFeatureExtractor

def reward(inputs, outputs):
    return torch.sum(inputs*outputs, dim=(1, 2, 3)) / torch.sum(((inputs+outputs)-inputs*outputs), dim=(1, 2, 3))


"""AQUI EMPIEZAN HIPERPARAMETROS. IMPLEMENTAR OPCIÓN DE SCRIPTING"""

parser = argparse.ArgumentParser()
parser.add_argument('-n_exp', '--num_experiment', type=int)

parser.add_argument('-s', '--num_steps', type=int)
parser.add_argument('-bs', '--batch_size', type=int)
parser.add_argument('-e', '--num_epochs', type=int)
parser.add_argument('-plr', '--policy_learning_rate', type=float)
parser.add_argument('-clr', '--critic_learning_rate', type=float)
parser.add_argument('-ep', '--entropy_param', type=float)

parser.add_argument('-sd', '--state_dicts', type=str)

args = parser.parse_args()

num_experiment = args.num_experiment if args.num_experiment is not None else 31
num_steps = args.num_steps if args.num_steps is not None else 10
batch_size = args.batch_size if args.batch_size is not None else 32
num_epochs = args.num_epochs if args.num_epochs is not None else 5
policy_learning_rate = args.policy_learning_rate if args.policy_learning_rate is not None else 0.0001
critic_learning_rate = args.critic_learning_rate if args.critic_learning_rate is not None else 0.001
entropy_param = args.entropy_param if args.entropy_param is not None else 0.0001
state_dicts_path = args.state_dicts


"""AQUI FINALIZAN LOS HIPERPARAMETROS"""

canvas_width = 64
grid_width = 32
use_color = False
brushes_basedir = '/home/albert/spiral/third_party/mypaint-brushes-1.3.0'
brush_type = 'classic/calligraphy'

environments = [Environment(canvas_width, grid_width, brush_type, use_color, brush_sizes=torch.linspace(1.2, 3, 10),
                          use_pressure=False, use_alpha=False, background="transparent", brushes_basedir=brushes_basedir)
                for i in range(batch_size)]

action_space_shapes = [environments[0]._action_spec[key].maximum+1 for key in environments[0]._action_spec]


dataset = MNIST("datasets", train=True, download=False, transform=transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x > 0,
    lambda x: x.float(),
]))

if state_dicts_path is not None:
    print("Retomando entrenamiento ya empezado")
    feature_extractor_critic = featureExtractors.ResNet12(in_chanels=2)
    feature_extractor_policy = featureExtractors.ResNet12(in_chanels=2)
    feature_extractor_critic.load_state_dict(torch.load("state_dicts/experiment"+str(num_experiment)+"_critic_feature_extractor")).cuda()
    feature_extractor_policy.load_state_dict(torch.load("state_dicts/experiment"+str(num_experiment)+"_policy_feature_extractor")).cuda()

    policy = policies.RNNPolicy(num_steps, action_space_shapes, feature_extractor_policy, lstm_size=512, batch_size=batch_size)
    policy.load_state_dict(torch.load("state_dicts/experiment"+str(num_experiment)+"_policy")).cuda()

    critic = critics.Critic(num_steps, feature_extractor_critic)
    critic.load_state_dict(torch.load("state_dicts/experiment"+str(num_experiment)+"_critic")).cuda()

else:
    print("Entrenando desde cero")
    feature_extractor = featureExtractors.ResNet12(in_chanels=2)
    feature_extractor = preTrainFeatureExtractor(feature_extractor, dataset, 64, 5, optimizer=Adam,
                                                 cuda=True)
    feature_extractor.pretraining = False
    feature_extractor_critic = feature_extractor
    feature_extractor_policy = copy.deepcopy(feature_extractor)

    policy = policies.RNNPolicy(num_steps, action_space_shapes, feature_extractor_policy, lstm_size=512,
                                batch_size=batch_size)
    critic = critics.Critic(num_steps, feature_extractor_critic)


discriminator = reward
trainRecurrent(environments, dataset, policy, critic, discriminator, num_steps, batch_size,
               num_epochs, policy_learning_rate, critic_learning_rate, entropy_param,
               optimizer=Adam, num_experiment=num_experiment)
