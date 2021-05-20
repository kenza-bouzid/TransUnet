from medpy import metric
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def calculate_metric_percase(pred, target):
    pred[pred > 0] = 1
    target[target > 0] = 1
    if pred.sum() > 0 and target.sum() > 0:
        dice = metric.binary.dc(pred, target)
        hd95 = metric.binary.hd95(pred, target)
        return (dice, hd95)
    elif pred.sum() > 0 and target.sum() == 0:
        return (1, 0)
    else:
        return (0, 0)


def evaluate_single_volume(image, label, model, classes=9):
    y_pred_vol = model.predict(image)
    y_pred_vol = tf.math.argmax(tf.nn.softmax(
        y_pred_vol, axis=-1), axis=-1).numpy()
    labels = tf.math.argmax(label, axis=-1).numpy()

    metric_list = []
    for i in range(1, classes):
        score = calculate_metric_percase(
            y_pred_vol == i, labels == i)
        metric_list.append(score)
    return metric_list


def inference(test_dataset, model, classes=9):
    metric_list = 0.0

    for data in tqdm(test_dataset):
        image = data['image']
        label = data['label']
        metric = evaluate_single_volume(image, label, model)
        metric_list += np.array(metric)

    print()
    metric_list /= len(test_dataset)
    for cls in range(1, classes):
        print(
            f"Class: {cls}, mean dice: {metric_list[cls-1][0]}, mean HD: {metric_list[cls-1][1]}")
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print(
        f'Testing performance in best val model: mean_dice : {performance} mean_hd95 : {mean_hd95}')


def inference_latex_table_row(test_dataset, model, classes=9):
    metric_list = 0.0

    for data in test_dataset:
        image = data['image']
        label = data['label']
        metric = evaluate_single_volume(image, label, model)
        metric_list += np.array(metric)

    print()
    metric_list /= len(test_dataset)
    print(f'{model.name}', end=' ')
    print(f'& {(np.mean(metric_list, axis=0)[0]*100):.2f}', end=' ')
    for cls in range(1, classes):
        print(
            f"& {(metric_list[cls-1][0]*100):.2f} ", end=' ')
    print('\\\\')
