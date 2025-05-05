import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
import random
import torch
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
os.makedirs("../src/Dataset/8Puzzle", exist_ok=True)

puzzle_bal, labels_bal = create_puzzle_dataset(20000, imbalance=False)
torch.save((puzzle_bal, labels_bal), "../src/Dataset/8Puzzle/8Puzzle_balanced.pt")

puzzle_imbal, labels_imbal = create_puzzle_dataset(20000, imbalance=True)
torch.save((puzzle_imbal, labels_imbal), "../src/Dataset/8Puzzle/8Puzzle_imbalanced.pt")

# print("Saved 8Puzzle datasets with correct 9-digit labels.")


#images, labels = torch.load("D:/From-Desktop/Hriday/BITS/4th year/Sem 2/AI/Project/AI Project/dataset/8Puzzle/8Puzzle_balanced.pt")
#print(images.shape)  # torch.Size([100, 84, 84])
#print(labels.shape)  # torch.Size([100, 9])
#print("First label (tile arrangement):", labels[0].tolist())


#import matplotlib.pyplot as plt

# Pick a sample index
#index = 0

# Get the image and label
#sample_img = images[index]
#sample_label = labels[index]

# Plot the image
#plt.imshow(sample_img, cmap='gray')
#plt.title(f"Label (placeholder): {sample_label}")
#plt.axis('off')
#plt.show()




