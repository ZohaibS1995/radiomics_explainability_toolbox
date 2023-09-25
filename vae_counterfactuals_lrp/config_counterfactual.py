import os
class Config:
    def __init__(self):
        # parameters for fold
        self.fold_no = 4
        self.batch_size = 4

        # model configurations
        self.input_channels = 3
        self.latent_dimension = 512
        self.patch_size = 224
        self.channels = (16, 32, 64, 128, 256, 512)
        self.strides = (1, 2, 2, 2, 2, 2)

        # model and checkpoint paths
        self.base_model_path = r'Z:\Xian\Models'
        self.model_name = 'VAE_MLP_only_image'
        self.vae_model_path = os.path.join(r"D:\Zohaib\SWE_images\models", f"4_checkpoint.pt")
        self.mlp_model_path = os.path.join(r"D:\Zohaib\SWE_images\models", f"0_mlp_checkpoint.pt")
        self.path_clinical_data = os.path.join(r'Z:\Xian\VAE', 'label_test.csv')

        # threshold for prediction
        self.threshold = 0.35

        # parameters for counterfactuals
        self.lambda_start = -4000
        self.lambda_end = 4000
        self.lambda_step = 400

        # Directories and Paths for data
        self.train_data_path = r"Z:\Xian\VAE\Train"
        self.test_data_path = r"Z:\Xian\VAE\Test"
        self.base_dir = r'Z:\Xian\VAE'

        # counterfactuals save directory
        self.counterfactuals_save_directory = r"D:\Zohaib\SWE_images\check_data\png"

config_counterfactuals = Config()
