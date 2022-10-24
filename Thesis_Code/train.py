from model import unet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger 
from Loss import calc_IOU, calc_IOU_loss
from dataset_prep import *
from Hyperparameter import *

#Hyperparameter
Hyperparameter=Hyperparameter()
#Dataset_ready
test_size = Hyperparameter['test_size'];
resized_shape = Hyperparameter['resized_shape'];
#Model
n_filter = Hyperparameter['n_filter'];
activation = Hyperparameter['activation'];
kernel_initializer = Hyperparameter['kernel_initializer'];
dropout=Hyperparameter['dropout']
output_activation=Hyperparameter['output_activation']
#Optimizer
learning_rate=Hyperparameter['learning_rate'];
decay_steps=Hyperparameter['decay_steps'];
decay_rate=Hyperparameter['decay_rate'];
beta_1=Hyperparameter['beta_1']
beta_2=Hyperparameter['beta_2']
epsilon=Hyperparameter['epsilon']
#Training
Batch_Size=Hyperparameter['Batch_Size'];
Epoch=Hyperparameter['Epoch'];
Workers=Hyperparameter['Workers'];

learning_schedule =ExponentialDecay(learning_rate,decay_steps=decay_steps,decay_rate=decay_rate,staircase=True)

#Create Model
Train_image, Train_mask, Test_image, Test_mask = input_data_ready(train_image_dir_l='C:/Users/s_t392/Downloads/Thesis/New_Image/labels/',
                     train_image_dir = 'C:/Users/s_t392/Downloads/Thesis/New_Image/images/',test_size=test_size, resized_shape=resized_shape)


model=unet(input_size = (resized_shape[0],resized_shape[1],3), n_filter = n_filter, activation = activation, dropout=dropout, 
           kernel_initializer = kernel_initializer, output_activation=output_activation)

model.compile(optimizer=Adam(learning_rate=learning_schedule,  beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
              loss=calc_IOU_loss, metrics=[calc_IOU]) 

#Callback
#earlystoping = EarlyStopping(monitor='loss', patience=10);
training_log= CSVLogger('training.log', separator=',', append=False);
callback_list=[training_log]

#Model_Train 
model.fit(x=Train_image/255,y=Train_mask,batch_size=Batch_Size,epochs=Epoch,validation_data=(Test_image/255, Test_mask),workers=Workers,callbacks=[callback_list])

##Save Model
model.save('Unet_model_5_30')