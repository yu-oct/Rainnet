import numpy as np
import matplotlib.pyplot as plt
from rainnet import rainnet  
import matplotlib as mpl
# Load the prepared input array 
X_input = np.load("X_input_cropped.npy")

# Initialize the RainNet model in regression mode with input shape matching your data
model = rainnet(input_shape=(128, 128, 4), mode="regression")

# Load the pretrained weights into the model
model.load_weights("rainnet_weights.h5")  

# Run prediction on the input sequence
y_pred = model.predict(X_input, verbose=1)

# Inverse the log-transform to recover rainfall in mm per 5 minutes
# Original transform: log(rain_mm + 0.01)
# Therefore, inverse: exp(y) - 0.01
rain_mm = np.exp(y_pred[0, :, :, 0]) - 0.01

Rt = np.exp(X_input[0,:,:,3])-0.01
Rt_5 = np.exp(X_input[0,:,:,2])-0.01
Rt_10 = np.exp(X_input[0,:,:,1])-0.01
Rt_15 = np.exp(X_input[0,:,:,0])-0.01

#Rt = X_input[0,:,:,3]
#Rt_5 = X_input[0,:,:,2]
#Rt_10 = X_input[0,:,:,1]
#Rt_15 = X_input[0,:,:,0]

cmted=plt.get_cmap('nipy_spectral', 512)
cmed=mpl.colors.ListedColormap(cmted(np.linspace(0.05, .95, 120)))

plt.figure(figsize=(5, 5))
plt.imshow(Rt_15, cmap=cmed, vmin=0, vmax=1.3)
plt.title("Rainfall (T-15)")
plt.colorbar(label="Rainfall (mm / 5 min)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(Rt_10, cmap="Blues", vmin=0, vmax=1.3)
plt.title("Rainfall (T-10)")
plt.colorbar(label="Rainfall (mm / 5 min)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(Rt_5, cmap="Blues", vmin=0, vmax=1.3)
plt.title("Rainfall (T-5)")
plt.colorbar(label="Rainfall (mm / 5 min)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(5, 5))
plt.imshow(Rt, cmap="Blues", vmin=0, vmax=1.3)
plt.title("Rainfall (T)")
plt.colorbar(label="Rainfall (mm / 5 min)")
plt.tight_layout()
plt.show()

# Plot the predicted rainfall map
plt.figure(figsize=(5, 5))
plt.imshow(rain_mm, cmap="Blues", vmin=0, vmax=1.3)
plt.title("Predicted Rainfall (T+5)")
plt.colorbar(label="Rainfall (mm / 5 min)")
plt.tight_layout()
plt.show()