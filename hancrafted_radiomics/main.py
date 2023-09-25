import os
import pickle
import joblib
import configparser
import pandas as pd

# local imports
from tab_shap import SHAPAnalysis
from tab_lime import LIMEAnalysis
from tab_dice import CounterFactualAnalysis


def load_model(file_path):
    """
    Load a machine learning model from different file formats.

    :param file_path: Path to the model file.
    :return: Loaded machine learning model.
    """
    if file_path.endswith('.sav'):
        # Load a scikit-learn model saved with joblib
        return joblib.load(file_path)


    elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        # Load a model saved with Python Pickle
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    elif file_path.endswith('.joblib'):
        # Load a model saved with joblib (scikit-learn)
        return joblib.load(file_path)

    else:
        raise ValueError("Unsupported model file format")

def read_config_file(file_path):
    """Parse a config file and return a configparser.ConfigParser object."""

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    # Instantiate a ConfigParser object and read the config file
    config = configparser.ConfigParser()
    config.read(file_path)

    return config

def main():
    """Main function to load model, test data, compute and plot SHAP values."""

    # Read and parse the config file
    config = read_config_file('config.ini')

    # Extract values from the config file
    # Check what time of explanations need to be generated
    generate_shap = config.get("DEFAULT", "SHAP", fallback=None)
    generate_lime = config.get("DEFAULT", "LIME", fallback=None)
    generate_counterfactuals = config.get("DEFAULT", "Counterfactuals", fallback=None)

    # get model file and test features file names
    model_filename_path = config.get("DEFAULT", "model_filename", fallback=None)
    test_features_path = config.get("DEFAULT", "test_features", fallback=None)

    # specify the directory name for saving the results
    save_dir_for_plots = config.get("DEFAULT", "save_dir_for_plots", fallback=None)

    # if plt_all is set to false, the explanations will be generated for the following instances
    local_plt_indices_list = [int(x) for x in config.get("DEFAULT", "local_plt_indices_list", fallback="").split(",")]

    # Shapley Additive Explanations Specific parameters
    top_n_dependence_plots = int(config.get("DEFAULT", "top_n_dependence_plots", fallback=0))
    top_n_dependence_interactions = int(config.get("DEFAULT", "top_n_dependence_interactions", fallback=0))
    class_names = config.get("DEFAULT", "class_names", fallback=0).split(",")
    local_plts_all_shap = bool(config.get("DEFAULT", "local_plt_all_shap", fallback=0) == "True")

    # counterfactuals specific parameters
    no_of_counterfactuals = int(config.get("DEFAULT", "no_of_counterfactuals", fallback=0))
    counterfactuals_plt_all = bool(config.get("DEFAULT", "counterfactuals_plt_all", fallback=0) == "True")

    # LIME specific parameters
    local_plts_all_lime = bool(config.get("DEFAULT", "local_plt_all_lime", fallback=0) == "True")

    # Load the trained model
    model = load_model(model_filename_path)

    # Load the test data
    df_test = pd.read_csv(test_features_path)
    feature_names = df_test.columns

    # ensuring that all the features present in dataframe are either int or float
    for col in feature_names[:-1]:
        if df_test[col].dtype is not float or df_test[col].dtype is not int:
            df_test[col] = df_test[col].astype(float)

    if generate_counterfactuals:
        counterfactual_analysis = CounterFactualAnalysis(model, save_dir_for_plots, feature_names=feature_names.to_list(), class_names=class_names)
        counterfactual_analysis.compute(df_test)
        if counterfactuals_plt_all:
            counterfactual_analysis.generate_counterfactuals_for_dataframe(df_test, num_cf=no_of_counterfactuals)
        else:
            counterfactual_analysis.generate_counterfactuals_for_indices(df_test, local_plt_indices_list, num_cf=no_of_counterfactuals)

    if generate_lime:
        lime_analysis = LIMEAnalysis(model, save_dir_for_plots, feature_names=feature_names, class_names=class_names)
        lime_analysis.compute(df_test)
        if local_plts_all_lime:
            lime_analysis.local_plot_dataframe()
        else:
            lime_analysis.local_plot(local_plt_indices_list)

    if generate_shap:
        # Initialize SHAPAnalysis with the trained model and compute SHAP values
        shap_analysis = SHAPAnalysis(model, save_dir_for_plots, feature_names=feature_names)
        shap_analysis.compute(df_test)

        # Generate global plots
        shap_analysis.global_summary_plot()
        shap_analysis.shap_dependence_plot(top_n_dependence_plots, no_of_interactions=top_n_dependence_interactions)

        # generate local plots
        if local_plts_all_shap:
            shap_analysis.local_plot_all()
        else:
            shap_analysis.local_plot(local_plt_indices_list)

if __name__ == "__main__":
    main()
