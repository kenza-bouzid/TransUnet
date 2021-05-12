import numpy as np
import matplotlib.pyplot as plt


def visualize(X, y, y_pred, sample_num, figsize=(20,20), cmap='viridis'):
    y_pred_np = y_pred.numpy()[sample_num,:,:,:]
    y_class = np.argmax(np.softmax(y_pred_np,axis=-1), axis=-1)
    x = X.numpy()[sample_num,:,:,:]
    y_np = y.numpy()[sample_num,:,:,:]
    y_np = np.argmax(y_np, axis=-1)

    fig, axis = plt.subplots(1, 2, figsize=figsize)
    axis[0,0].imshow(x, cmap='gray') 
    axis[0,0].imshow(y_np, cmap=cmap, alpha=0.3) 
    axis[0,0].set_title("original labels")
    axis[0,1].imshow(x, cmap='gray') 
    axis[0,1].imshow(y_class, cmap=cmap, alpha=0.3)
    axis[0,1].set_title("predicted labels")
    plt.show()

