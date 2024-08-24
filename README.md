# mlproject
## Project Components

1. **Data Ingestion** (`src/components/data_ingestion.py`): Handles the process of collecting and importing data into the project.

2. **Data Transformation** (`src/components/data_transformation.py`): Preprocesses and transforms the raw data into a format suitable for model training.

3. **Model Trainer** (`src/components/model_trainer.py`): Implements the machine learning model and training process.

4. **Prediction Pipeline** (`src/pipeline/predict_pipeline.py`): Manages the flow of data through the trained model for making predictions.

5. **Training Pipeline** (`src/pipeline/train_pipeline.py`): Orchestrates the entire training process from data ingestion to model evaluation.

6. **Exception Handling** (`src/exception.py`): Custom exception handling for better error management.

7. **Logging** (`src/logger.py`): Implements logging functionality for tracking the execution of various components.

8. **Utilities** (`src/utils.py`): Contains utility functions used across the project.

## Notebooks

The `notebook` directory contains Jupyter notebooks for exploratory data analysis (EDA) and initial model training:

- `1. EDA STUDENT PERFORMANCE.ipynb`: Exploratory Data Analysis of the student performance dataset.
- `2. MODEL TRAINING.ipynb`: Initial model training experiments.

## Getting Started
# ## Environment Setup

This project uses Conda for environment management. Follow these steps to set up your environment:

1. Install Anaconda or Miniconda if you haven't already.

2. Create a new Conda environment: ``conda create -n mlproject python=3.8
3.  Activate the environment:  ``conda activate mlproject
   <br>
   Or if using pip:

  ``pip install -r requirements.txt <br>
1. Clone this repository.
2. Install the required dependencies (provide requirements.txt or environment setup instructions).
3. Run the training pipeline: `python src/pipeline/train_pipeline.py`
4. Make predictions using the prediction pipeline: `python src/pipeline/predict_pipeline.py`

## License

This project is licensed under the [MIT].
