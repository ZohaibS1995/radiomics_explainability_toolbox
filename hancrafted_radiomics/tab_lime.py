# Import necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import base64
from lime.lime_tabular import LimeTabularExplainer


class LIMEAnalysis:
    def __init__(self, model, store_folder_path="lime_results", feature_names=None, class_names=None):
        self.model = model
        self.explainer = None
        self.feature_names = feature_names[:-1]
        self.target_names = feature_names[-1]
        self.class_names = class_names
        self.folder_path = store_folder_path
        self.no_of_features = None
        self.path_local = os.path.join(self.folder_path, "local_plots_lime")
        os.makedirs(self.path_local, exist_ok=True)

    def compute(self, X, mode='classification'):
        if X is None or not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('Input data must be a Pandas DataFrame or a Numpy array.')

        training_data = X.values if isinstance(X, pd.DataFrame) else X
        training_data = training_data[:, :-1]

        # Initialize LIME explainer
        self.explainer = LimeTabularExplainer(training_data,
                                              feature_names=self.feature_names,
                                              class_names=self.class_names,
                                              mode=mode)
        self.data = training_data
        self.no_of_features = len(self.feature_names)

    def local_plot(self, instance_indices, filename="local_plot"):
        if self.explainer is None:
            raise ValueError('Compute method must be called before generating plots.')

        if not isinstance(instance_indices, list) or any(i < 0 or i >= len(self.data) for i in instance_indices):
            raise ValueError('Instance indices must be a list of valid indices.')

        for j, index in enumerate(instance_indices):
            instance = self.data.iloc[index].values if isinstance(self.data, pd.DataFrame) else self.data[index]
            exp = self.explainer.explain_instance(instance, self.model.predict_proba,
                                                  num_features=self.no_of_features)

            fig = exp.as_pyplot_figure()

            if False:
                if self.class_names and len(self.class_names) > 2:  # If it's multiclass
                    # Save plots for each class with its name in the filename
                    for idx, class_name in enumerate(self.class_names):
                        fig = exp.as_pyplot_figure(label=idx)
                        plt.savefig(os.path.join(self.path_local, f"{filename}_instance_{j}_class_{class_name}.png"),
                                    bbox_inches='tight')
                        plt.clf()
            if self.class_names and len(self.class_names) > 2:
                plt.savefig(os.path.join(self.path_local, f"{filename}_instance_{j}.png"), bbox_inches='tight')
                plt.close()
            else:
                # Save the generated plot for binary classification
                plt.savefig(os.path.join(self.path_local, f"{filename}_instance_{j}.png"), bbox_inches='tight')
                plt.close()

    def local_plot_dataframe(self, filename="local_plot"):
        if self.explainer is None:
            raise ValueError('Compute method must be called before generating plots.')

        for j in range(self.data.shape[0]):
            instance = self.data[j]
            exp = self.explainer.explain_instance(instance, self.model.predict_proba,
                                                  num_features=self.no_of_features)

            fig = exp.as_pyplot_figure()

            if False:
                if self.class_names and len(self.class_names) > 2:  # If it's multiclass
                    # Save plots for each class with its name in the filename
                    for idx, class_name in enumerate(self.class_names):
                        fig = exp.as_pyplot_figure(label=idx)
                        plt.savefig(os.path.join(self.path_local, f"{filename}_instance_{j}_class_{class_name}.png"),
                                    bbox_inches='tight')
                        plt.clf()
            if self.class_names and len(self.class_names) > 2:
                plt.savefig(os.path.join(self.path_local, f"{filename}_instance_{j}.png"), bbox_inches='tight')
                plt.close()
            else:
                # Save the generated plot for binary classification
                plt.savefig(os.path.join(self.path_local, f"{filename}_instance_{j}.png"), bbox_inches='tight')
                plt.close()

    def encode_image_base64(self, filename):
        with open(filename, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def create_image_html_str(self, img_encoded, title):
        return f'<h3>{title}</h3>\n<img src="data:image/png;base64,{img_encoded}">\n<br/>\n'

    def write_html(self, filename, html_strs, main_title):
        with open(filename, 'w') as f:
            f.write(f"<html>\n<head>\n<title>{main_title}</title>\n</head>\n<body>\n")
            for html_str in html_strs:
                f.write(html_str)
            f.write("</body>\n</html>")

