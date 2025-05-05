#!/usr/bin/env python3
# 8Puzzle_LLM_Solver.py
# Solution for CS F407 Artificial Intelligence Project Assignment II
# Exercise 2a: End-to-end 8-Puzzle solver using Neural Network and LLM
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from groq import Groq
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchvision import datasets, transforms

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("../src/Dataset",exist_ok=True)
mnist_full = datasets.MNIST(root="../src/Dataset",train=True, download=True, transform= transforms.ToTensor())

x_train = mnist_full.data.numpy()
y_train = mnist_full.targets.numpy()

np.savez_compressed("../src/Dataset/mnist_data.npz",x_train=x_train, y_train=y_train)

data = np.load("../src/Dataset/mnist_data.npz")
x_train = data['x_train']
y_train = data['y_train']

# Organize MNIST digits
digit_images = {i: [] for i in range(10)}
for img, label in zip(x_train, y_train):
    digit_images[label].append(Image.fromarray(img))

def generate_puzzle_tensor(index, imbalance=False):
    numbers = list(range(9))
    if imbalance and random.random() < 0.5:
        numbers.remove(0)
        numbers.insert(4, 0)  # Force 0 at center
    random.shuffle(numbers)
    
    canvas = Image.new("L", (28 * 3, 28 * 3))
    
    for idx, num in enumerate(numbers):
        row, col = divmod(idx, 3)
        digit_img = random.choice(digit_images[num]).resize((28, 28))
        canvas.paste(digit_img, (col * 28, row * 28))
    
    img_tensor = torch.tensor(np.array(canvas), dtype=torch.float32) / 255.0
    label_tensor = torch.tensor(numbers, dtype=torch.long)
    return img_tensor, label_tensor

def create_puzzle_dataset(num_samples=100, imbalance=False):
    images, labels = [], []
    for i in range(num_samples):
        img, lbl = generate_puzzle_tensor(i, imbalance)
        images.append(img)
        labels.append(lbl)
    return torch.stack(images), torch.stack(labels)

# Create and save datasets
os.makedirs("../src/Datasetllm", exist_ok=True)

puzzle_bal, labels_bal = create_puzzle_dataset(1, imbalance=False)
torch.save((puzzle_bal), "../src/Datasetllm/8Puzzle1.pt")
puzzle_bal1 = create_puzzle_dataset(1, imbalance=False)
torch.save((puzzle_bal1), "../src/Datasetllm/8Puzzle2.pt")
with open("../src/config8puz.json", "r") as f:
    config = json.load(f)

model_params = config["model_params"]

# Initialize Groq client - replace with your API key
client = Groq(api_key="")  # Enter your API key here

class Puzzle8LLMSolver:
    """
    End-to-end 8-Puzzle solver using Neural Network for state recognition
    and LLM for generating solution steps
    """
    
    def __init__(self, inputsize,activation,hiddenlayers,outputsize):
        super(Puzzle8LLMSolver, self).__init__()
        layers = []
        input_size = inputsize
        def get_activation(name):
            return {
                    "relu": nn.ReLU(),
                    "tanh": nn.Tanh(),
                    "sigmoid": nn.Sigmoid()
                    }.get(name.lower(), nn.ReLU())  # default to ReLU
        activation = get_activation(activation)
        for hidden_size in hiddenlayers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation)
            input_size = hidden_size

        layers.append(nn.Linear(input_size,outputsize))
        self.model = nn.Sequential(*layers)

    def forward(self ,x):
        x = x.view(x.size(0), -1)
        return self.model(x)
    
    def load_model(self, model_path):
        """Load the trained model from checkpoint"""
        try:
            # Load the model architecture - assuming you've already implemented this in Exercise 1
            model = Puzzle8LLMSolver(model_params["input_size"],model_params["activation"],model_params["hidden_layers"],model_params["output_size"])
            model = torch.load(model_path)
            model.eval()  # Set to evaluation mode
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image_path):
        """Preprocess the image for the neural network"""
        try:
            # Load and preprocess the image - this should match your Exercise 1 preprocessing
            image = torch.load(image_path)
            sample = image[0].numpy().reshape(84,84)
            # Define the grid size (3x3)
            grid_size = 3
            height, width = sample.shape[0], sample.shape[1]

            partition_height = height // grid_size
            partition_width = width // grid_size
            partitioned_images = []
            for i in range(grid_size):
                for j in range(grid_size):
                # Slice the tensor into smaller parts
                    partition = sample[i*partition_height:(i+1)*partition_height, j*partition_width:(j+1)*partition_width]
                    partitioned_images.append(partition)
            for i in range (9):
                partitioned_images[i]= torch.tensor(partitioned_images[i], dtype=torch.float32).view(-1).unsqueeze(0)
            
            return image.to(self.device)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            sys.exit(1)
    
    def recognize_puzzle_state(self, image_path):
        """Recognize the puzzle state from an image using the trained neural network"""
        # Preprocess the image
        image = self.preprocess_image(image_path)
        
        # Pass through the model to get predictions
        predicted_classes = []
        for i in range (9):
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_classes[i] = torch.max(probabilities, 1)
        
        puzzle_state = predicted_classes.cpu().numpy().reshape(3, 3)
        
        return puzzle_state
    
    def format_puzzle_state(self, state):
        """Format the puzzle state as a string for LLM prompt"""
        state_str = ""
        for row in state:
            state_str += " ".join(str(cell) for cell in row) + "\n"
        return state_str
    
    def prompt_llm(self, source_state, goal_state):
        """
        Prompt the LLM to reason about the 8-Puzzle solution
        Returns intermediate states of the solution
        """
        # Format the states for the prompt
        source_str = self.format_puzzle_state(source_state)
        goal_str = self.format_puzzle_state(goal_state)
        
        # Create a prompt for the LLM
        prompt = f"""
        You are an expert 8-Puzzle solver. The 8-Puzzle consists of a 3x3 grid with 8 numbered tiles (1-8) and one empty space.
        You can slide tiles into the empty space to rearrange the puzzle.
        
        The current state of the puzzle is:
        {source_str}
        
        The goal state is:
        {goal_str}
        
        Please provide the sequence of intermediate states to solve this puzzle. 
        Start with the source state and end with the goal state.
        For each step, only one tile can move into the adjacent empty space.
        
        Format your response as a JSON array of states, where each state is a 3x3 grid.
        For example:
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 0]],
            [[1, 2, 3], [4, 5, 0], [7, 8, 6]],
            ...
        ]
        
        The number 0 represents the empty space.
        """
        
        try:
            # Call LLM API to get the solution
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert 8-Puzzle problem solver. Provide step-by-step solutions in JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.3-70b-versatile",  # Adjust based on available model
                temperature=0.2,  # Lower temperature for more deterministic outputs
                response_format={"type": "json_object"}  # Request JSON formatted response
            )
            
            # Extract the response content
            solution = response.choices[0].message.content.strip()
            
            # Parse the JSON solution
            try:
                solution_data = json.loads(solution)
                return solution_data
            except json.JSONDecodeError:
                print("Error: LLM response is not valid JSON. Attempting to extract JSON...")
                # Try to extract JSON from the response if it contains additional text
                import re
                json_match = re.search(r'\[\s*\[.*\]\s*\]', solution, re.DOTALL)
                if json_match:
                    try:
                        solution_data = json.loads(json_match.group(0))
                        return solution_data
                    except:
                        print("Failed to extract JSON from response.")
                
                print("Raw response:", solution)
                return None
            
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return None
    
    def solve_puzzle(self, source_image_path, goal_image_path):
        """
        Solve the 8-Puzzle by:
        1. Recognizing the source and goal states from images
        2. Prompting the LLM for a solution
        3. Saving the solution states to a JSON file
        """
        print("Recognizing source state...")
        source_state = self.recognize_puzzle_state(source_image_path)
        print(f"Source state recognized: \n{self.format_puzzle_state(source_state)}")
        
        print("Recognizing goal state...")
        goal_state = self.recognize_puzzle_state(goal_image_path)
        print(f"Goal state recognized: \n{self.format_puzzle_state(goal_state)}")
        
        print("Prompting LLM for solution...")
        solution_states = self.prompt_llm(source_state, goal_state)
        
        if solution_states:
            print("Solution found! Saving to states.json...")
            with open("states.json", "w") as f:
                json.dump(solution_states, f, indent=4)
            print("Solution saved successfully!")
            return solution_states
        else:
            print("Failed to generate a solution.")
            return None

def main():
    """Main function to parse arguments and run the solver"""
    parser = argparse.ArgumentParser(description='8-Puzzle Solver using Neural Network and LLM')
    parser.add_argument('--model', type=str, required=True, help='../src/modular.pth')
    parser.add_argument('--source', type=str, required=True, help='../src/Datasetllm/8Puzzle1.pt')
    parser.add_argument('--goal', type=str, required=True, help='../src/Datasetllm/8Puzzle2.pt')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist")
        sys.exit(1)
    if not os.path.exists(args.source):
        print(f"Error: Source image {args.source} does not exist")
        sys.exit(1)
    if not os.path.exists(args.goal):
        print(f"Error: Goal image {args.goal} does not exist")
        sys.exit(1)
    
    # Initialize and run the solver
    solver = Puzzle8LLMSolver(args.model)
    solver.solve_puzzle(args.source, args.goal)

if __name__ == "__main__":
    main()
