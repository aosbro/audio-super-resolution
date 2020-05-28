from utils.constants_parser import get_general_args
from trainers.wgan_trainer import WGanTrainer
from utils.utils import prepare_maestro_data
import argparse
import os


def get_wgan_trainer_args():
    """
    Parses the arguments related to the training of the gan if provided by the user, otherwise uses default values.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Trains the GAN.')
    # Data related constants
    parser.add_argument('--use_npy', default=True, type=bool,
                        help='Flag indicating if the data is stored as multiple .npy files or a single .hdf5 file.')
    parser.add_argument('--hdf5_filepath', type=str, help='Location of the .hdf5 file if this data format is selected.')
    parser.add_argument('--train_npy_filepath', default='data/train.npy', type=str,
                        help='Location of the train .npy file if this data format is selected.')
    parser.add_argument('--test_npy_filepath', default='data/test.npy', type=str,
                        help='Location of the test .npy file if this data format is selected.')
    parser.add_argument('--valid_npy_filepath', default='data/valid.npy', type=str,
                        help='Location of the valid .npy file if this data format is selected.')
    parser.add_argument('--train_batch_size', default=64, type=int,
                        help='Number of samples per batch during the train phase.')
    parser.add_argument('--test_batch_size', default=64, type=int,
                        help='Number of samples per batch during the test phase.')
    parser.add_argument('--valid_batch_size', default=64, type=int,
                        help='Number of samples per batch during the validation phase.')
    parser.add_argument('--train_shuffle', default=True, type=bool,
                        help='Flag indicating if the train data must be shuffled.')
    parser.add_argument('--test_shuffle', default=True, type=bool,
                        help='Flag indicating if the test data must be shuffled.')
    parser.add_argument('--valid_shuffle', default=True, type=bool,
                        help='Flag indicating if the validation data must be shuffled.')
    parser.add_argument('--num_worker', default=2, type=int, help='Number of workers used by the data loaders.')

    # Trainer related constants
    parser.add_argument('--savepath', default='/content/drive/My Drive/audio_objects/gan_trainer.tar', type=str,
                        help='Location where to save the gan trainer to resume training.')
    parser.add_argument('--loadpath', default='', type=str,
                        help='Location of an existing gan trainer from which to resume training.')
    parser.add_argument('--lambda_adversarial', default=1e-4, type=float,
                        help='Weight given to the adversarial loss during the GAN training.')
    parser.add_argument('--lambda_time', default=1.0, type=float,
                        help='Weight given to the l2 loss in dime domain during the GAN training.')
    parser.add_argument('--use_penalty', default=False, type=bool,
                        help='Flag indicating whether to use gradient penalty or weight clipping to enforce the '
                             'Lipschitz constraint on the discriminator.')
    parser.add_argument('--clipping_limit', default=0.01, type=float,
                        help='Maximum absolute value for the weights of the discriminator.')
    parser.add_argument('--gamma_wgan_gp', default=10, type=float,
                        help='Weight given to the gradient penalty in the discriminator loss.')
    parser.add_argument('--n_critic', default=1, type=int,
                        help='Number of discriminator update before doing a single generator update.')
    parser.add_argument('--coupling_epoch', default=0, type=int,
                        help='Epoch from which the generator receives the feedback from the discriminator.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train the models on.')

    # Generator related constants
    parser.add_argument('--generator_path', default=None, type=str,
                        help='Location of a pre-trained generator used to generate the samples. If not provided, a new '
                             'generator will be instantiated and trained from scratch. Providing a pre-trained '
                             'generator can help stabilizing the training.')
    parser.add_argument('--generator_lr', default=1e-3, type=float, help='Learning rate for the generator.')
    parser.add_argument('--generator_scheduler_step', default=30, type=int,
                        help='Number of steps before the learning step is reduced by a factor gamma.')
    parser.add_argument('--generator_scheduler_gamma', default=0.5, type=float,
                        help='Factor by which the learning rate is reduced after a specified number of steps.')

    # Discriminator related constants
    parser.add_argument('--discriminator_lr', default=1e-3, type=float, help='Learning rate for the discriminator.')
    parser.add_argument('--discriminator_scheduler_step', default=30, type=int,
                        help='Number of steps before the learning step is reduced by a factor gamma.')
    parser.add_argument('--discriminator_scheduler_gamma', default=0.5, type=float,
                        help='Factor by which the learning rate is reduced after a specified number of steps.')
    args = parser.parse_args()
    return args


def get_wgan_trainer(general_args, trainer_args):
    """
    Instantiates the GanTrainer class based on the given arguments.
    :param general_args: instance of an argument parser that stores generic parameters.
    :param trainer_args: instance of an argument parser that stores parameters related to the training.
    :return: instance of an GanTrainer.
    """
    train_loader, _, valid_loader = prepare_maestro_data(trainer_args)

    # Load the train class which will automatically resume previous state from 'loadpath'
    wgan_trainer = WGanTrainer(train_loader=train_loader,
                               test_loader=None,
                               valid_loader=valid_loader,
                               general_args=general_args,
                               trainer_args=trainer_args)
    return wgan_trainer


if __name__ == '__main__':
    # TODO: Delete next line
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Get the parameters related to the track generation
    trainer_args = get_wgan_trainer_args()

    # Get the general parameters
    general_args = get_general_args()

    # Get the trainer
    gan_trainer = get_wgan_trainer(general_args, trainer_args)
    gan_trainer.train(epochs=trainer_args.epochs)
