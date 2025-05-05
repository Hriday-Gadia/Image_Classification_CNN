# # # /////Hands-on exrcise 1-b
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from PIL import Image
import random
import os
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


# Organize digits
digit_images = {i: [] for i in range(10)}
for img, label in zip(x_train, y_train):
    digit_images[label].append(Image.fromarray(img))

# Country codes (2 digits)
COUNTRY_CODES = ["91", "44", "33", "49", "81", "01", "61", "86", "39", "34"]

def generate_phone_tensor(index, imbalance=False):
    # Generate 10-digit number with optional imbalance
    if imbalance and random.random() < 0.5:
        digit = str(random.randint(0, 9))
        number = ''.join([digit if random.random() < 0.4 else str(random.randint(0, 9)) for _ in range(10)])
    else:
        number = ''.join([str(random.randint(0, 9)) for _ in range(10)])

    code = random.choice(COUNTRY_CODES)
    full_number = code + number  # 12 digits total
    label = [int(d) for d in full_number]

    canvas = Image.new("L", (12 * 28, 28))
    for i, digit in enumerate(label):
        digit_img = random.choice(digit_images[digit]).resize((28, 28))
        canvas.paste(digit_img, (i * 28, 0))

    img_tensor = torch.tensor(np.array(canvas), dtype=torch.float32) / 255.0
    label_tensor = torch.tensor(label, dtype=torch.long)
    return img_tensor, label_tensor

def create_phone_dataset(num_samples=100, imbalance=False):
    images, labels = [], []
    for i in range(num_samples):
        img, lbl = generate_phone_tensor(i, imbalance)
        images.append(img)
        labels.append(lbl)
    return torch.stack(images), torch.stack(labels)

# Save datasets
os.makedirs("../src/Dataset/PhoneNum", exist_ok=True)

phone_bal, labels_bal = create_phone_dataset(20000, imbalance=False)
torch.save((phone_bal, labels_bal), "../src/Dataset/PhoneNum/PhoneNum_balanced.pt")

phone_imbal, labels_imbal = create_phone_dataset(20000, imbalance=True)
torch.save((phone_imbal, labels_imbal), "../src/Dataset/PhoneNum/PhoneNum_imbalanced.pt" )

# print("Saved PhoneNum datasets with correct 12-digit labels.")

#images, labels = torch.load("D:/From-Desktop/Hriday/BITS/4th year/Sem 2/AI/Project/AI Project/dataset/PhoneNum/PhoneNum_balanced.pt")
#print(images.shape)   # torch.Size([100, 28, 336])
#print(labels.shape)   # torch.Size([100, 12])
#print("First label (phone number):", labels[0].tolist())


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


