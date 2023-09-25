# Handcrafted Radiomics Explanations Tool

## Introduction

The Handcrafted Radiomics Explanations Tool is a Python-based utility for generating explanations using three different techniques: LIME (Local Interpretable Model-agnostic Explanations), SHAP (SHapley Additive exPlanations), and DICE (Counterfactual Explanations). This tool is designed to provide insights into radiomics models and enhance the interpretability of their predictions.

## Getting Started

Before you can use the Radiomics Explanations Tool, you'll need to prepare your environment and configuration:

### Prerequisites

- Install the required Python packages (specified in `requirements.txt`)

### Configuration

1. **Model Configuration:** Ensure that you have a machine learning model ready for interpretation. The tool supports models saved in various formats, including `.sav`, `.pkl`, and `.joblib`. Update the `model_filename` field in the `config.ini` file with the path to your model file.

2. **Test Data:** Prepare your test data in a CSV format with features as columns. Update the `test_features` field in the `config.ini` file with the path to your test data file. Please append a last column named `target` containing the labels in the test dataframe.

3. **Configuration File:** Customize the tool's behavior by editing the `config.ini` file. You can specify which explanations you want to generate (LIME, SHAP, DICE), configure visualization options, and set parameters for each explanation technique.

## Explanation Generation

To generate explanations for your radiomics model, follow these steps:

1. Run the `main()` function in the main script (`main.py`):

   ```bash
   python main.py
   ```

2. The tool will read the configuration from `config.ini` and load the specified model and test data.

3. Explanations will be generated based on your configuration settings. The tool supports the following:

   - **SHAP:** SHAP values are computed and visualized to explain global and local model behavior.

   - **LIME:** LIME explanations are generated to provide insights into the model's predictions at a local level.

   - **DICE:** Counterfactual explanations are generated to show how changes in input features can lead to different predictions.

4. Explanations will be saved in the specified directory (`save_dir_for_plots`) in various formats, including plots and files.

## Customization

You can customize the tool's behavior by editing the `config.ini` file. Here are some key customization options:

- `SHAP`, `LIME`, and `Counterfactuals`: Set these options to `"True"` to enable the corresponding explanation methods.

- `local_plt_indices_list`: Specify the indices of instances for which you want to generate local explanations.

- Customize various parameters for SHAP, LIME, and DICE, such as the number of counterfactuals, the number of dependence plots, and class names.

## Results

Generated explanations will be saved in the specified directory (`save_dir_for_plots`). You can find both global and local plots, allowing you to gain insights into the model's behavior and predictions.

## Conclusion

The Radiomics Explanations Tool provides a convenient way to enhance the interpretability of your radiomics model. By generating LIME, SHAP, and DICE explanations, you can gain valuable insights into how your model makes predictions, both globally and for specific instances.


**Note:** Ensure that you have proper permissions to read the model and test data files and write to the output directory specified in `config.ini`.