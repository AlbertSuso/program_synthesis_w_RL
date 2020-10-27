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
from TrainingProcedures.training import train, preTrainFeatureExtractor

def mse_per_batch(inputs, outputs):
    return torch.sum(inputs*outputs, dim=(1, 2)) / torch.sum(((inputs+outputs)-inputs*outputs), dim=(1, 2))


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

num_experiment = args.num_experiment if args.num_experiment is not None else 0
num_steps = args.num_steps if args.num_steps is not None else 5
batch_size = args.batch_size if args.batch_size is not None else 32
num_epochs = args.num_epochs if args.num_epochs is not None else 2
policy_learning_rate = args.policy_learning_rate if args.policy_learning_rate is not None else 0.001
critic_learning_rate = args.critic_learning_rate if args.critic_learning_rate is not None else 0.001
entropy_param = args.entropy_param if args.entropy_param is not None else 0.001


"""AQUI FINALIZAN LOS HIPERPARAMETROS"""

canvas_width = 28
grid_width = 28
use_color = False
brushes_basedir = '/home/albert/spiral/third_party/mypaint-brushes-1.3.0'
brush_type = 'classic/calligraphy'

environments = [Environment(canvas_width, grid_width, brush_type, use_color, brush_sizes=torch.linspace(1, 3, 20),
                          use_pressure=True, use_alpha=False, background="transparent", brushes_basedir=brushes_basedir)
                for i in range(batch_size)]


dataset = MNIST("/data2fast/users/asuso", train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    lambda x: x > 0,
    lambda x: x.float(),
]))

feature_extractor = featureExtractors.ResNet12(in_chanels=1)
feature_extractor.cuda()
policy = policies.MnistPolicy()
policy.cuda()
critic = critics.Critic()
critic.cuda()
discriminator = mse_per_batch


preTrainFeatureExtractor(feature_extractor, dataset, 64, 3, optimizer=Adam, cuda=True)

train(environments, dataset, feature_extractor, copy.deepcopy(feature_extractor), policy, critic, discriminator, num_steps, batch_size, num_epochs, policy_learning_rate, critic_learning_rate, entropy_param, optimizer=Adam, num_experiment=num_experiment)
