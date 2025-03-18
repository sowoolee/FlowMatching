import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic 2D data
n_samples = 3000
mu1, mu2 = [2, 2], [-2, -2]
sigma = 0.1 * np.eye(2)

data1 = np.random.multivariate_normal(mu1, sigma, n_samples)
data2 = np.random.multivariate_normal(mu2, sigma, n_samples)
data = np.vstack([data1, data2])

# Shuffle data indices
idx = np.random.permutation(len(data))
train_ratio = 0.8
n_train = int(train_ratio * len(idx))
train_idx, val_idx = idx[:n_train], idx[n_train:]

data_train, data_val = data[train_idx], data[val_idx]

# Convert to tensors
data_train_tensor = torch.tensor(data_train, dtype=torch.float32)
data_val_tensor = torch.tensor(data_val, dtype=torch.float32)

# Base noise
z_train = torch.randn_like(data_train_tensor)
z_val = torch.randn_like(data_val_tensor)

# Define MLP network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
net = MLP()

# Loss function (Flow Matching Loss)
def flow_matching_loss(net, data, z):
    n = data.shape[0]
    t = torch.rand(n, 1, dtype=data.dtype, device=data.device)  # Sample time t
    x_t = t * data + (1 - t) * z  # Interpolation between data and noise
    v_target = data - z  # Target velocity

    t_expanded = t.expand(-1, 1)
    x_input = torch.cat([x_t, t_expanded], dim=1)  # Concatenating t for input
    v_pred = net(x_input)  # Predict velocity

    loss = ((v_pred - v_target) ** 2).mean()

    # x1_pred = (1 - t) * v_pred + x_t
    # projection_loss = ((x1_pred[:, 1] - x1_pred[:, 0]) ** 2).mean()
    # loss += projection_loss * 10.

    return loss

# Training parameters
num_epochs = 2000
learning_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

train_loss_history = []
val_loss_history = []

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss = flow_matching_loss(net, data_train_tensor, z_train)
    loss.backward()
    optimizer.step()

    train_loss_history.append(loss.item())

    with torch.no_grad():
        val_loss = flow_matching_loss(net, data_val_tensor, z_val)
        val_loss_history.append(val_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# RK4 Sampling function
def RK4_sampling(net, z_init, num_steps=10, return_flows=False):
    inter_zs = []
    h = 1.0 / num_steps  # Step size
    z = torch.tensor(z_init, dtype=torch.float32)
    t = torch.zeros(z.shape[0], 1, dtype=torch.float32)

    for _ in range(num_steps):
        k1 = net(torch.cat([z, t], dim=1))
        k2 = net(torch.cat([z + (h / 2) * k1, t], dim=1))
        k3 = net(torch.cat([z + (h / 2) * k2, t], dim=1))
        k4 = net(torch.cat([z + h * k3, t], dim=1))
        z = z + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

        z_optim = z.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z_optim], lr=0.1)
        opt_steps = 1
        for step in range(opt_steps):
            optimizer.zero_grad()
            loss = ((z_optim[:, 1] - z_optim[:, 0]) ** 2).mean()
            loss.backward()
            optimizer.step()
        # z = z_optim.detach()

        inter_zs.append(z.detach().numpy())

        t += h

    if return_flows:
        return inter_zs
    else:
        return z.detach().numpy()

# Generate new noise samples
z_new = np.random.randn(500, 2).astype(np.float32)
generated_flows = RK4_sampling(net, z_new, num_steps=100, return_flows=True)
generated_data = generated_flows[-1]

generated_data_projected = generated_data.copy()
generated_data_projected[:, 0] = (generated_data[:, 0] + generated_data[:, 1]) / 2
generated_data_projected[:, 1] = (generated_data[:, 0] + generated_data[:, 1]) / 2

# Create velocity field for visualization
x_range = np.linspace(-3, 3, 20)
y_range = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x_range, y_range)
grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)

# Convert to tensor and add time component t=0.5
grid_t = 0.5 * np.ones((grid_points.shape[0], 1))
grid_input = np.hstack([grid_points, grid_t])
grid_tensor = torch.tensor(grid_input, dtype=torch.float32)

# Predict velocity vectors
with torch.no_grad():
    V = net(grid_tensor).numpy()

Ux = V[:, 0].reshape(X.shape)  # x-component of velocity
Uy = V[:, 1].reshape(Y.shape)  # y-component of velocity

generated_data_torch = torch.tensor(generated_data, dtype=torch.float32, requires_grad=True)
optimizer = optim.Adam([generated_data_torch], lr=0.1)
num_steps = 100
for step in range(num_steps):
    optimizer.zero_grad()
    loss = ((generated_data_torch[:, 1] - generated_data_torch[:, 0]) ** 2).mean()
    loss.backward()
    optimizer.step()
generated_data_optimized = generated_data_torch.detach().numpy()


plt.figure(figsize=(20, 6))

# scatter plot of data
plt.subplot(2, 5, 1)
plt.scatter(data[:, 0], data[:, 1], s=15)
plt.title('Synthetic 2D Gaussian Mixture Data')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

# plot training loss
plt.subplot(2, 5, 2)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Flow Matching Training Loss')
plt.legend()

# plot generated sample
plt.subplot(2, 5, 3)
plt.scatter(generated_data[:, 0], generated_data[:, 1], s=15)
plt.title('Generated Samples via RK4 100 steps')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

# plot velocity field
plt.subplot(2, 5, 4)
plt.quiver(X, Y, Ux, Uy, scale=50, color='blue')
plt.title('Learned Velocity Field')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.subplot(2, 5, 5)
plt.scatter(generated_data_projected[:, 0], generated_data_projected[:, 1], s=15)
plt.title('Projected Samples')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

for i, step in enumerate([25, 50, 75]):
    plt.subplot(2, 5, i + 6)
    plt.scatter(generated_flows[step][:, 0], generated_flows[step][:, 1], s=15)
    plt.title(f'Step {step}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

plt.subplot(2, 5, 9)
plt.scatter(generated_data_optimized[:, 0], generated_data_optimized[:, 1], s=15)
plt.title('Optimized Samples (during inference)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.subplot(2, 5, 10)
plt.scatter(generated_data_optimized[:, 0], generated_data_optimized[:, 1], s=15)
plt.title('Optimized Samples (after inference)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

plt.tight_layout()
plt.show()