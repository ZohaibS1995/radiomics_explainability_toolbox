import pandas as pd
import numpy as np
import os
from dice_ml import Dice
from dice_ml.utils import helpers  # for loading dataset
import dice_ml
import json

class CounterFactualAnalysis:
    def __init__(self, model, store_folder_path="dice_results",feature_names=None, class_names=None):
        self.model = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')
        self.dice_explainer = None
        self.data_interface = None
        self.folder_path = store_folder_path
        self.feature_names = feature_names
        self.class_names = class_names
        self.path_local = os.path.join(self.folder_path, "counterfactuals")
        os.makedirs(self.path_local, exist_ok=True)

    def compute(self, data_interface):
        # Initialize DiCE explainer
        self.data_interface = dice_ml.Data(dataframe= data_interface, continuous_features=self.feature_names[:-1], outcome_name="target")
        self.dice_explainer = Dice(self.data_interface, self.model, method="random")


    def generate_counterfactuals_for_dataframe(self, df, num_cf=1):
        if self.dice_explainer is None:
            raise ValueError('Compute method must be called before generating counterfactuals.')

        # Loop over the dataframe
        for index in range(len(df)):
            print("Idx: ", index)

            # Generating counterfactuals for the current instance
            dice_result = self.dice_explainer.generate_counterfactuals(df.iloc[[index], :-1], total_CFs=num_cf)
            # Save the counterfactual instances for the current instance

            json_result_str = dice_result.to_json()
            json_result = json.loads(json_result_str)
            test_data = np.squeeze(json_result["test_data"])[None, ...]
            feature_names = np.squeeze(json_result["feature_names_including_target"])
            cfs_arr = np.vstack(json_result["cfs_list"])
            data_arr = np.concatenate((test_data, cfs_arr), axis=0)
            df_result = pd.DataFrame(data_arr, columns=feature_names)
            df_result.to_csv(os.path.join(self.path_local, f"counterfactual_instance_{index}.csv"))


    def generate_counterfactuals_for_indices(self, df, local_plt_indices_list, num_cf=1):
        if self.dice_explainer is None:
            raise ValueError('Compute method must be called before generating counterfactuals.')

        # Loop over the dataframe
        for index in local_plt_indices_list:
            print("Idx: ", index)

            # Generating counterfactuals for the current instance
            dice_result = self.dice_explainer.generate_counterfactuals(df.iloc[[index], :-1], total_CFs=num_cf)
            # Save the counterfactual instances for the current instance

            json_result_str = dice_result.to_json()
            json_result = json.loads(json_result_str)
            test_data = np.squeeze(json_result["test_data"])[None, ...]
            feature_names = np.squeeze(json_result["feature_names_including_target"])
            cfs_arr = np.vstack(json_result["cfs_list"])
            data_arr = np.concatenate((test_data, cfs_arr), axis=0)
            df_result = pd.DataFrame(data_arr, columns=feature_names)
            df_result.to_csv(os.path.join(self.path_local, f"counterfactual_instance_{index}.csv"))