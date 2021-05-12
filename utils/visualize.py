import numpy as np
import matplotlib.pyplot as plt


def visualize(image, label, model, figsize=(20,20), cmap='viridis'):
    y_pred = model.predict(image).numpy()
    y_pred = np.argmax(y_pred, axis=-1)
    y = np.argmax(label, axis=-1)
    fig, axis = plt.subplots(1, 2, figsize=figsize)
    axis[0,0].imshow(image, cmap='gray') 
    axis[0,0].imshow(y, cmap=cmap, alpha=0.3) 
    axis[0,1].imshow(image, cmap='gray') 
    axis[0,1].imshow(y_pred, cmap=cmap, alpha=0.3)
    plt.show()

