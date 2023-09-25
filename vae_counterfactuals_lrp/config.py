import os

class Config:
    def __init__(self):
        # parameters for fold
        self.fold_no = 4
        self.batch_size = 4


        # VAE training parameters
        self.gpu_use = True
        self.patience_early_stopping = 30
        self.input_channels = 3
        self.latent_dimension = 512
        self.patch_size = 224
        self.channels = (16, 32, 64, 128, 256, 512)
        self.strides = (1, 2, 2, 2, 2, 2)

        # VAE training params
        self.vae_learning_rate = 1e-4
        self.vae_patience_early_stopping=10

        # epochs and patience
        self.n_epochs = 20
        self.patience_early_stopping = 5

        # Directories and Paths
        self.train_data_path = r"Z:\Xian\VAE\Train"
        self.test_data_path = r"Z:\Xian\VAE\Test"
        self.base_dir = r'Z:\Xian\VAE'

        # CUDA Configuration
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = "4"

        # Data splits
        self.k_splits = 5
        self.shuffle_data = True
        self.random_seed = 3

        # paths for storing training data
        self.check_data_snippets = r"D:\Zohaib\SWE_images\check_data\png"
        self.check_data_test_snippets = r"D:\Zohaib\SWE_images\check_data\pngs_test"
        self.model_save_dir = r"D:\Zohaib\SWE_images\models"


        # vae-gan autoencoder training params
        self.adv_weight = 0.01
        self.perceptual_weight = 0.001
        self.kl_weight = 1e-6
        self.autoencoder_warm_up_n_epochs = 5


    def print_config(self):
        print("Configuration:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

config = Config()
#config.print_config()