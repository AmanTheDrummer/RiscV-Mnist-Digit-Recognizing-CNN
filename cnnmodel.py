# This is the Python implementation on which we created a CNN with 8 3x3 convolutional filters.
# Layer1: 784 nodes (input layer) -> 10 nodes (hidden layer)
# Layer2: 10 nodes (hidden layer) -> 10 nodes (output layer)
# Activation function ReLU on Hidden Layer
# Activation function softmax on Output Layer

import numpy as np
import pandas as pd

# --- Activation Functions ---
def ReLU(Z): return np.maximum(0, Z)
def dReLU(Z): return Z > 0
def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / np.sum(expZ)

# --- Data Loading ---
def load_image(row):
    img = row[1:].values.reshape(28, 28) / 255.0
    label = int(row.iloc[0])
    return img, label

# --- CNN Layers ---
def convolve2d(img, filters):
    n_filters, f_h, f_w = filters.shape
    h, w = img.shape
    output = np.zeros((n_filters, h - f_h + 1, w - f_w + 1))
    for n in range(n_filters):
        for i in range(h - f_h + 1):
            for j in range(w - f_w + 1):
                output[n, i, j] = np.sum(img[i:i+f_h, j:j+f_w] * filters[n])
    return output

def maxpool(img, size=2, stride=2):
    c, h, w = img.shape
    out = np.zeros((c, h // size, w // size))
    for k in range(c):
        for i in range(0, h, size):
            for j in range(0, w, size):
                out[k, i // size, j // size] = np.max(img[k, i:i+size, j:j+size])
    return out

# --- Initialization ---
def init_params():
    conv_filters = np.random.randn(8, 3, 3) * 0.1
    W1 = np.random.randn(100, 8 * 13 * 13) * np.sqrt(2 / (8 * 13 * 13))
    b1 = np.zeros((100, 1))
    W2 = np.random.randn(10, 100) * np.sqrt(2 / 100)
    b2 = np.zeros((10, 1))
    return conv_filters, W1, b1, W2, b2

# --- Forward Pass ---
def forward(img, conv_filters, W1, b1, W2, b2):
    conv = convolve2d(img, conv_filters)
    relu = ReLU(conv)
    pooled = maxpool(relu)
    flat = pooled.flatten().reshape(-1, 1)
    Z1 = W1 @ flat + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = softmax(Z2)
    return A2, flat, A1, Z1, pooled, relu, conv

# --- Loss and Accuracy ---
def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-9)

def predict(output): return np.argmax(output)

# --- Derivative of Softmax-CrossEntropy Combined ---
def d_softmax_crossentropy(pred, label):
    grad = pred.copy()
    grad[label] -= 1
    return grad

# --- Backward Pass ---
def backward(img, label, conv_filters, W1, b1, W2, b2, A2, flat, A1, Z1, pooled, relu, conv, lr=0.01):
    # Output Layer Gradients
    dZ2 = d_softmax_crossentropy(A2, label)
    dW2 = dZ2 @ A1.T
    db2 = dZ2

    # Hidden Layer Gradients
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * dReLU(Z1)
    dW1 = dZ1 @ flat.T
    db1 = dZ1

    # Update Fully Connected Weights
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # Backprop through maxpool and ReLU (partial, simplified for demo)
    d_flat = W1.T @ dZ1
    d_pool = d_flat.reshape(pooled.shape)

    d_relu = np.zeros_like(relu)
    c, h, w = relu.shape
    for k in range(c):
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                pool_region = relu[k, i:i+2, j:j+2]
                max_val = np.max(pool_region)
                for m in range(2):
                    for n in range(2):
                        if i + m < h and j + n < w and relu[k, i+m, j+n] == max_val:
                            d_relu[k, i+m, j+n] = d_pool[k, i//2, j//2]

    d_conv = d_relu * dReLU(conv)

    # Gradient for conv filters
    d_filters = np.zeros_like(conv_filters)
    for n in range(conv_filters.shape[0]):
        for i in range(conv.shape[1]):
            for j in range(conv.shape[2]):
                region = img[i:i+3, j:j+3]
                d_filters[n] += d_conv[n, i, j] * region

    # Update conv filters
    conv_filters -= lr * d_filters

    return conv_filters, W1, b1, W2, b2

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv(r"C:\Users\Amans\Desktop\SAB3R\My Workspace\MNIST Digit Recognizer\MNIST Digit Recognizer\train.csv")
    print("Initializing parameters...")
    conv_filters, W1, b1, W2, b2 = init_params()

    epochs = 10
    lr = 0.001

    print("Starting training...\n")
    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1}/{epochs} ---")
        total_loss = 0
        correct = 0

        for i in range(1000):  # Using 1000 samples for demo
            try:
                img, label = load_image(df.iloc[i])
                A2, flat, A1, Z1, pooled, relu, conv = forward(img, conv_filters, W1, b1, W2, b2)
                loss = cross_entropy(A2, label)
                total_loss += loss
                correct += (predict(A2) == label)

                conv_filters, W1, b1, W2, b2 = backward(
                    img, label, conv_filters, W1, b1, W2, b2,
                    A2, flat, A1, Z1, pooled, relu, conv, lr
                )

                if (i + 1) % 200 == 0:
                    print(f"Processed {i + 1}/1000 samples")

            except Exception as e:
                print(f"Error at sample {i}: {e}")
                break

        print(f"Epoch {epoch + 1} completed.")
        print(f"Loss: {float(total_loss) / 1000:.4f} | Accuracy: {correct / 1000 * 100:.2f}%\n")

    print("Saving trained parameters to text files...")

    # Save convolution filters
    with open("conv_filters.txt", "w") as f:
        for filt in conv_filters:
            for row in filt:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("\n")  # Separate filters by a newline

    # Save weights and biases
    np.savetxt("W1.txt", W1)
    np.savetxt("b1.txt", b1)
    np.savetxt("W2.txt", W2)
    np.savetxt("b2.txt", b2)

    print("Parameter saving complete.")
