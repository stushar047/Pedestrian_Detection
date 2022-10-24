from tensorflow.keras.layers import Conv2D,MaxPooling2D,Add,Activation,Input,Dropout,concatenate,UpSampling2D
from tensorflow.keras import Model

def unet(input_size = (256,256,1), n_filter = 8, activation = 'relu',kernel_initializer = 'he_normal', dropout=0.5, output_activation='sigmoid'):
    inputs = Input(input_size, name='INPUT')
    conv1 = Conv2D(n_filter, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='CONV1_1')(inputs)
    conv1 = Conv2D(n_filter, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='CONV1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='MAX_1')(conv1)
    conv2 = Conv2D(n_filter*2, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='CONV2_1')(pool1)
    conv2 = Conv2D(n_filter*2, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='CONV2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='MAX_2')(conv2)
    conv3 = Conv2D(n_filter*4, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='CONV3_1')(pool2)
    conv3 = Conv2D(n_filter*4, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='CONV3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='MAX_3')(conv3)
    conv4 = Conv2D(n_filter*8, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='CONV4_1')(pool3)
    conv4 = Conv2D(n_filter*8, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer,name='CONV4_2')(conv4)
    drop4 = Dropout(dropout, name='Dropout_4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='MAX_4')(drop4)

    conv5 = Conv2D(n_filter*16, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='CONV5_1')(pool4)
    conv5 = Conv2D(n_filter*16, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='CONV5_2')(conv5)
    drop5 = Dropout(dropout, name='Dropout_5')(conv5)

    up6 = Conv2D(n_filter*8, 2, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='ConvConcat4')(UpSampling2D(size = (2,2), name='UpSample4')(drop5))
    merge6 = concatenate([drop4,up6], axis = 3, name='Concat4')
    conv6 = Conv2D(n_filter*8, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name= 'DeConv4_1')(merge6)
    conv6 = Conv2D(n_filter*8, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name= 'DeConv4_2')(conv6)

    up7 = Conv2D(n_filter*4, 2, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name= 'ConvConcat3')(UpSampling2D(size = (2,2), name= 'UpSample3')(conv6))
    merge7 = concatenate([conv3,up7], axis = 3, name='Concat3')
    conv7 = Conv2D(n_filter*4, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='DeConv3_1')(merge7)
    conv7 = Conv2D(n_filter*4, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='DeConv3_2')(conv7)

    up8 = Conv2D(n_filter*2, 2, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name= 'ConvConcat2')(UpSampling2D(size = (2,2), name= 'UpSample2')(conv7))
    merge8 = concatenate([conv2,up8], axis = 3, name='Concat2')
    conv8 = Conv2D(n_filter*2, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='DeConv2_1')(merge8)
    conv8 = Conv2D(n_filter*2, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='DeConv2_2')(conv8)

    up9 = Conv2D(n_filter, 2, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='ConvConcat1')(UpSampling2D(size = (2,2), name='UpSample1')(conv8))
    merge9 = concatenate([conv1,up9], axis = 3, name='Concat1')
    conv9 = Conv2D(n_filter, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='DeConv1_1')(merge9)
    conv9 = Conv2D(n_filter, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='DeConv1_2')(conv9)
    #conv9 = Conv2D(n_filter, 3, activation = activation, padding = 'same', kernel_initializer = kernel_initializer, name='DeConv1_3')(conv9)
    conv10 = Conv2D(1, 1, activation = output_activation, name='output')(conv9)

    model = Model(inputs, conv10)

    return model