# Import necessary libraries
import shap
import pandas as pd
import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
import matplotlib.pyplot as plt
import base64
import os

# Define SHAPAnalysis class
class SHAPAnalysis:
    # Initialize with a machine learning model
    def __init__(self, model, store_folder_path = "shap_results", feature_names = None):
        # Machine Learning model
        self.model = model   
        
        # SHAP explainer object, to be initialized later
        self.explainer = None   

        # SHAP values object, will be computed later
        self.shap_values = None  

        # Expected values of the model prediction
        self.expected_value = None   

        # Flag to determine if the problem is multi-class or not
        self.is_multiclass = False   

        # Dataset on which SHAP values will be computed
        self.data = None

        # Write the feature names here
        self.feature_names = feature_names[:-1]
        self.target_names = feature_names[-1]

        # Paths for storing the information. 
        self.folder_path = store_folder_path
        self.path_global = os.path.join(self.folder_path, "shap_global_summary_plots")
        self.path_local = os.path.join(self.folder_path, "shap_local_plots")
        self.path_dependence = os.path.join(self.folder_path, "shap_dependence_plots")

        os.makedirs(self.path_global, exist_ok=True)
        os.makedirs(self.path_local, exist_ok=True)
        os.makedirs(self.path_dependence, exist_ok=True)

    # Compute SHAP values using the given dataset
    def compute(self, X):
        """
        Initialize explainer and compute SHAP values
        """
        # Validate the input dataset
        if X is None or not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError('Input data must be a Pandas DataFrame or a Numpy array.')

        X = X.values if isinstance(X, pd.DataFrame) else X
        X = X[:, :-1]

        # Initialize SHAP explainer with the model and input dataset
        if self.feature_names is not None:
            self.explainer = shap.Explainer(self.model, X, feature_names=self.feature_names)
        else:
            self.explainer = shap.Explainer(self.model, X, feature_names=self.feature_names)

        # Compute SHAP values for the given dataset
        try:
            self.shap_values = self.explainer(X)
        except:
            self.shap_values = self.explainer(X, check_additivity=False)

        # Compute the expected value of the model predictions
        self.expected_value = self.explainer.expected_value

        # Store the input dataset
        self.data = X

        # Check if the problem is multiclass or not based on the shape of SHAP values

        if len(self.shap_values.shape) == 2:
            self.is_multiclass = False
        elif self.shap_values.shape[-1] > 2:
            self.is_multiclass = True
        else:
            self.is_multiclass = False
            self.shap_values = self.shap_values[...,1]

    # Generate and save global summary plot
    def global_summary_plot(self, filename="summary_plot.png"):
        """
        Generate global summary plot for SHAP values and save it
        """
        # Initialize HTML file and list for storing HTML strings
        html_file = "summary.html"
        html_strs = []

        # Ensure that SHAP values have been computed
        if self.shap_values is None:
            raise ValueError('Compute method must be called before generating plots.')

        # Add title for the HTML page
        html_strs.append('<h1>Global Summary Plot</h1>\n')

        # If the problem is multiclass, generate and save a summary plot for each class
        if self.is_multiclass:
            for idx in range(self.shap_values.shape[2]):
                # Generate SHAP summary plot
                shap.plots.beeswarm(self.shap_values[...,idx], show=False, max_display = 24)

                # Save the generated plot
                plt.savefig(os.path.join(self.path_global, f"{filename}_class_{idx}.png"), bbox_inches='tight')

                # Clear the plot for the next one
                plt.close()

                # Convert the saved plot into a base64 encoded string for embedding in HTML
                img_encoded = self.encode_image_base64(os.path.join(self.path_global, f"{filename}_class_{idx}.png"))

                # Append HTML string for each plot
                html_strs.append(self.create_image_html_str(img_encoded, f"Class {idx}"))

        # If the problem is not multiclass, generate and save only one summary plot
        else:
            # Generate SHAP summary plot
            shap.plots.beeswarm(self.shap_values, show=False, max_display = 24)

            # Save the generated plot
            plt.savefig(os.path.join(self.path_global, filename + ".png"),  bbox_inches='tight')

            # Clear the plot for the next one
            plt.close()

            # Convert the saved plot into a base64 encoded string for embedding in HTML
            img_encoded = self.encode_image_base64(os.path.join(self.path_global, filename + ".png"))

            # Append HTML string for the plot
            html_strs.append(self.create_image_html_str(img_encoded, ""))

        # Write all HTML strings to a single HTML file
        self.write_html(os.path.join(self.path_global, html_file), html_strs, main_title="Global Summary Plots")

    # Generate and save local explanation plot
    def local_plot(self, instance_indices, filename="local_plot.png"):
        """
        Generate local explanation plot for selected instances and save it
        """

        # Ensure that SHAP values have been computed

        if self.shap_values is None:
            raise ValueError('Compute method must be called before generating plots.')
        
        # Validate instance indices
        if not isinstance(instance_indices, list) or any(i < 0 or i >= len(self.shap_values) for i in instance_indices):
            raise ValueError('Instance indices must be a list of valid indices.')
        
        # Initialize HTML file and list for storing HTML strings
        html_file = "local.html"
        html_strs = []

        # Add title for the HTML page
        html_strs.append('<h1>Local Plots</h1>\n')

        # Generate plots for multiclass scenario
        if self.is_multiclass:
            for idx in range(self.shap_values.shape[2]):
                for j, index in enumerate(instance_indices):
                    # Generate local plot for each instance
                    shap.waterfall_plot(self.shap_values[...,idx][index], show=False, max_display = 24)

                    # Save the generated plot
                    plt.savefig(os.path.join(self.path_local, f"{filename}_class_{idx}_instance_{j}.png"), bbox_inches='tight')

                    # Clear the plot for the next one
                    plt.close()

                    # Convert the saved plot into a base64 encoded string for embedding in HTML
                    img_encoded = self.encode_image_base64(os.path.join(self.path_local, f"{filename}_class_{idx}_instance_{j}.png"))

                    # Append HTML string for each plot
                    html_strs.append(self.create_image_html_str(img_encoded, f"Class {idx} Instance {j}"))
        else:
            for j, index in enumerate(instance_indices):
                # Generate local plot for each instance
                shap.waterfall_plot(self.shap_values[index], show=False, max_display = 24)

                # Save the generated plot
                plt.savefig(os.path.join( self.path_local, f"{filename}_instance_{j}.png"), bbox_inches='tight')

                # Clear the plot for the next one
                plt.close()

                # Convert the saved plot into a base64 encoded string for embedding in HTML
                img_encoded = self.encode_image_base64(os.path.join( self.path_local, f"{filename}_instance_{j}.png"))

                # Append HTML string for each plot
                html_strs.append(self.create_image_html_str(img_encoded, f"Instance {j}"))

        # Write all HTML strings to a single HTML file
        self.write_html(os.path.join(self.path_local, html_file), html_strs, main_title="Local Plots")

    def local_plot_all(self, filename="local_plot"):
        """
        Generate local explanation plot for selected instances and save it
        """

        # Ensure that SHAP values have been computed

        if self.shap_values is None:
            raise ValueError('Compute method must be called before generating plots.')


        # Initialize HTML file and list for storing HTML strings
        html_file = "local.html"
        html_strs = []

        # Add title for the HTML page
        html_strs.append('<h1>Local Plots</h1>\n')

        # Generate plots for multiclass scenario
        if self.is_multiclass:
            for idx in range(self.shap_values.shape[2]):
                for index in range(self.data.shape[0]):
                    # Generate local plot for each instance
                    shap.waterfall_plot(self.shap_values[..., idx][index], show=False, max_display=24)

                    # Save the generated plot
                    plt.savefig(os.path.join(self.path_local, f"{filename}_class_{idx}_instance_{index}.png"),
                                bbox_inches='tight')

                    # Clear the plot for the next one
                    plt.close()

                    # Convert the saved plot into a base64 encoded string for embedding in HTML
                    img_encoded = self.encode_image_base64(
                        os.path.join(self.path_local, f"{filename}_class_{idx}_instance_{index}.png"))

                    # Append HTML string for each plot
                    html_strs.append(self.create_image_html_str(img_encoded, f"Class {idx} Instance {index}"))
        else:
            for index in range(self.data.shape[0]):
                # Generate local plot for each instance
                shap.waterfall_plot(self.shap_values[index], show=False, max_display=24)

                # Save the generated plot
                plt.savefig(os.path.join(self.path_local, f"{filename}_instance_{index}.png"), bbox_inches='tight')

                # Clear the plot for the next one
                plt.close()

                # Convert the saved plot into a base64 encoded string for embedding in HTML
                img_encoded = self.encode_image_base64(os.path.join(self.path_local, f"{filename}_instance_{index}.png"))

                # Append HTML string for each plot
                html_strs.append(self.create_image_html_str(img_encoded, f"Instance {index}"))

        # Write all HTML strings to a single HTML file
        self.write_html(os.path.join(self.path_local, html_file), html_strs, main_title="Local Plots")


    # Function to generate SHAP dependence plots
    def shap_dependence_plot(self, top_n_features, no_of_interactions = 1, filename="dependence_plot"):
        """
        Generate SHAP dependence plots for features and save as images.

        Parameters:
        top_n_features (int): The number of top features to create dependence plots for.
        no_of_interactions (int, optional): The number of interactions to consider. Defaults to 1.
        filename (str, optional): The base filename to save the plots. Defaults to "dependence_plot".

        Raises:
        ValueError: If SHAP values have not been computed, or the number of features is not a positive integer.

        Outputs:
        Creates and saves image files of SHAP dependence plots.
        """
        # Raise an error if SHAP values have not been computed
        if self.shap_values is None:
            raise ValueError('Compute method must be called before generating plots.')
        
        # Raise an error if the number of features is not a positive integer
        if top_n_features <= 0 or not isinstance(top_n_features, int):
            raise ValueError('The number of features must be a positive integer.')
        
        html_file = "dependence.html"
        html_strs = []

        # If the problem is multiclass, generate dependence plots for each class
        if self.is_multiclass:
            # For each class
            for idx in range(self.shap_values.shape[2]):
                html_strs.append(f'<h2>Class {idx}</h2>\n')
                # Find the indices of the top features
                top_features = np.argsort(np.abs(self.shap_values[...,idx].values).mean(0))[:-top_n_features-1:-1]
                # For each top feature
                for j, feature in enumerate(top_features):
                    html_strs.append(f'<h3>Feature {j}</h3>\n')
                    # If there are interactions to consider
                    if no_of_interactions > 0:
                        # Find the indices of the interacting features
                        interactions = shap.approximate_interactions(feature, self.shap_values[...,idx].values, self.data)[:no_of_interactions]
                        # For each interaction
                        for k, interaction in enumerate(interactions):
                            # Generate dependence plot considering interaction
                            shap.dependence_plot(feature, self.shap_values[...,idx].values, self.data, interaction_index=interaction, show=False, feature_names=self.feature_names)
                            # Save the plot to a file
                            plt.savefig(os.path.join(self.path_dependence, f"{filename}_class_{idx}_feature_{j}_interaction_{k}.png"), bbox_inches='tight')
                            plt.close()
                            # Encode the image for HTML
                            img_encoded = self.encode_image_base64(os.path.join(self.path_dependence, f"{filename}_class_{idx}_feature_{j}_interaction_{k}.png"))
                            # Create HTML string for the plot and append to the list
                            html_strs.append(self.create_image_html_str(img_encoded, f"Interaction {k} Dependence Plot"))

        # If the problem is not multiclass, generate dependence plots without considering class
        else:
            # Find the indices of the top features
            top_features = np.argsort(np.abs(self.shap_values.values).mean(0))[:-top_n_features-1:-1]
            # For each top feature
            for j, feature in enumerate(top_features):
                html_strs.append(f'<h2>Feature {j}</h2>\n')
                # If there are interactions to consider
                if no_of_interactions > 0:
                    # Find the indices of the interacting features

                    interactions = shap.approximate_interactions(feature, self.shap_values.values, self.data)[:no_of_interactions]
                    # For each interaction
                    for k, interaction in enumerate(interactions):
                        # Generate dependence plot considering interaction
                        shap.dependence_plot(feature, self.shap_values.values, self.data, interaction_index=interaction, show=False, feature_names=self.feature_names)
                        # Save the plot to a file
                        plt.savefig(os.path.join(self.path_dependence, f"{filename}_feature_{j}_interaction_{k}.png"), bbox_inches='tight')
                        plt.close()
                        # Encode the image for HTML
                        img_encoded = self.encode_image_base64(os.path.join(self.path_dependence, f"{filename}_feature_{j}_interaction_{k}.png"))
                        # Create HTML string for the plot and append to the list
                        html_strs.append(self.create_image_html_str(img_encoded, f"Interaction {k} Dependence Plot"))

        # Write the HTML strings to a file
        self.write_html(os.path.join(self.path_dependence, html_file), html_strs, main_title="Dependence Plots")

    # Function to encode an image to base64 format
    def encode_image_base64(self, filename):
        """
        Encodes an image to base64 format.

        Parameters:
        filename (str): The name of the file to be encoded.

        Returns:
        str: A base64 encoded PNG image as a string.
        """
        # Open the image file in binary mode and read the contents
        with open(filename, "rb") as img_file:
            # Convert the binary image data to base64 string and return
            return base64.b64encode(img_file.read()).decode('utf-8')

    # Function to create HTML string for an image
    def create_image_html_str(self, img_encoded, title):
        """
        Creates a HTML string to embed an image.

        Parameters:
        img_encoded (str): The base64 encoded image string.
        title (str): The title of the image.

        Returns:
        str: A HTML string that embeds the image.
        """
        # Format and return the HTML string
        return f'<h3>{title}</h3>\n<img src="data:image/png;base64,{img_encoded}">\n<br/>\n'

    # Function to write HTML strings to a file
    def write_html(self, filename, html_strs, main_title):
        """
        Writes a collection of HTML strings to a file.

        Parameters:
        filename (str): The name of the HTML file.
        html_strs (List[str]): A list of HTML strings to write to the file.
        main_title (str): The main title for the HTML file.
        """
        # Open the file in write mode
        with open(filename, 'w') as f:
            # Write the HTML header with the title
            f.write(f"<html>\n<head>\n<title>{main_title}</title>\n</head>\n<body>\n")
            # Write each HTML string to the file
            for html_str in html_strs:
                f.write(html_str)
            # Write the HTML footer
            f.write("</body>\n</html>")
