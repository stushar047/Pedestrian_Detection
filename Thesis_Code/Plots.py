#Training and Valiadtion Graph
from pandas import read_csv
import matplotlib.pyplot as plt

Data=read_csv('training.log');
epoch=Data['epoch']
calc_IOU_loss=Data['calc_IOU']
loss=Data['loss']
val_calc_IOU_loss=Data['val_calc_IOU']
val_loss=Data['val_loss']

plt.plot(loss)
plt.plot(val_loss)
plt.title('model IOU Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()