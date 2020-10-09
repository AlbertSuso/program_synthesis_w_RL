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
    return torch.sum((inputs-outputs)**2, dim=(1, 2, 3)) - torch.sum((inputs+outputs) == 2, dim=(1, 2, 3))


"""AQUI EMPIEZAN HIPERPARAMETROS. IMPLEMENTAR OPCIÓN DE SCRIPTING"""

num_steps = 5
batch_size = 32
num_epochs = 20


"""AQUI FINALIZAN LOS HIPERPARAMETROS"""

canvas_width = 28
grid_width = 14
use_color = False
brushes_basedir = '/home/albert/spiral/third_party/mypaint-brushes-1.3.0'
brush_type = 'classic/calligraphy'

environments = [Environment(canvas_width, grid_width, brush_type, use_color, brush_sizes=torch.linspace(1, 3, 20),
                          use_pressure=True, use_alpha=False, background="transparent", brushes_basedir=brushes_basedir)
                for i in range(batch_size)]


dataset = MNIST("datasets", train=False, download=True, transform=transforms.ToTensor())


feature_extractor = featureExtractors.ResNet12(in_chanels=1)
feature_extractor.cuda()
policy = policies.MnistPolicy()
policy.cuda()
critic = critics.Critic()
critic.cuda()
discriminator = mse_per_batch


preTrainFeatureExtractor(feature_extractor, dataset, 64, 3, optimizer=Adam, cuda=True)

train(environments, dataset, feature_extractor, policy, critic, discriminator, num_steps, batch_size, num_epochs, optimizer=Adam, entropy_param=500)