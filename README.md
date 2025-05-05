# CSF407_2025_2021B4AA2794H
Question 1: 
MLP Neural Network for Image Classification
Design and train two separate multi-layer perceptron (MLP) neural networks:

One to classify image representations of 8-puzzle game states.

Another to classify images of phone numbers.
Each model is trained on both balanced and imbalanced datasets. The objective is to analyze the performance using accuracy, precision, recall, and visualize the results for both data distributions.
Setup Instructions
This project includes a config.yml file that defines all dependencies. To create a Conda environment with all required packages, run:

        conda env create -f config.yml

Once the environment is created, activate it.

- **Scripts**: `8Puzzle_data.py`, `8Puzzle_model.py`, `8Puzzle_trainer.py`
- **Generated Dataset Path**: `src/Dataset/8puzzle_balanced.pt`, `8puzzle_imbalanced.pt`
- **Results**: Plotted accuracy, precision, recall stored in `results/`

 Note:
-Keep track of the file paths and ensure that you have the permission to creatte files in the given file path.
-In Config, ensure that input size should be 784 as each individual digit image is 28x28
(No output graphs have been provided, as running 30 epoch of 20,000 images was proven diffiult owing to the local processor)



Question 2:8-Puzzle State Reasoning with Neural Tags and LLM
Use the trained 8-puzzle classifier to predict tags from image states and guide an LLM-based simulator 

- **Script**: `Exercise2a_main.py`
- **Input Arguments**:
  - `--model`: Path to MLP model (e.g. `src/Checkpoints/8puzzle_mlp.pt`)
  - `--source`: Image path of initial puzzle state (from test set)
  - `--goal`: Image path of goal puzzle state
- **Output**:
  - A file named `states.json` logging each intermediate state as returned by LLM based on neural tag predictions

Note: Ensure to put in GROQ API key into the LLM model
