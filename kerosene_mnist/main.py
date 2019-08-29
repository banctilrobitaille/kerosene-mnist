import logging

import torchvision
from kerosene.config.parsers import YamlConfigurationParser
from kerosene.events import Event
from kerosene.events.handlers.console import ConsoleLogger
from kerosene.events.handlers.visdom.config import VisdomConfiguration
from kerosene.events.handlers.visdom.visdom import VisdomLogger
from kerosene.events.preprocessors.visdom import PlotAllModelStateVariables
from kerosene.training.trainers import ModelTrainerFactory, SimpleTrainer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from kerosene_mnist.models import SimpleNet

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"

    model_trainer_config, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    train_loader = DataLoader(torchvision.datasets.MNIST('/files/', train=True, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_train, shuffle=True)

    test_loader = DataLoader(torchvision.datasets.MNIST('/files/', train=False, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_valid, shuffle=True)

    # Initialize the loggers
    visdom_logger = VisdomLogger(VisdomConfiguration.from_yml(CONFIG_FILE_PATH))

    # Initialize the model trainers
    model_trainer = ModelTrainerFactory(model=SimpleNet()).create(model_trainer_config)

    # Train with the training strategy
    trainer = SimpleTrainer("MNIST Trainer", train_loader, test_loader, model_trainer) \
        .with_event_handler(ConsoleLogger(), Event.ON_EPOCH_END) \
        .with_event_handler(visdom_logger, Event.ON_EPOCH_END, PlotAllModelStateVariables()) \
        .train(training_config.nb_epochs)
