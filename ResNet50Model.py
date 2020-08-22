from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D
from keras.models import Model,load_model

class ResNet50(object):

    def ResNet_IdentityBlock(Input_X, f, filters, stage, block):
    
        conv_name_base= 'RES' + str(stage) + block + '_branch'
        bn_name_base= 'BN' + str(stage) + block + '_branch'

        F1,F2,F3=filters
        Shortcut_X=Input_X

        Input_X=Conv2D(
                filters=F1,
                kernel_size=(1,1),
                strides=(1,1),
                padding="valid",
                name=conv_name_base+'2a',
                kernel_initializer="glorot_uniform"
        )(Input_X)
        Input_X=BatchNormalization(
            axis=3,
            name=bn_name_base+'2a'
        )(Input_X)
        Input_X=Activation('relu')(Input_X)

        Input_X=Conv2D(
            filters=F2,
            kernel_size=(f,f),
            strides=(1,1),
            padding="same",
            name=conv_name_base+'2b',
            kernel_initializer="glorot_uniform"
        )(Input_X)
        Input_X=BatchNormalization(
            axis=3,
            name=bn_name_base+'2b'
        )(Input_X)
        Input_X=Activation('relu')(Input_X)

        Input_X=Conv2D(
            filters=F3,
            kernel_size=(1,1),
            strides=(1,1),
            padding="valid",
            name=conv_name_base+'2c',
            kernel_initializer="glorot_uniform"
        )(Input_X)
        Input_X=BatchNormalization(
            axis=3,
            name=bn_name_base+'2c'
        )(Input_X)

        Input_X=Add()([Shortcut_X,Input_X])
        Input_X=Activation('relu')(Input_X)

        return Input_X

    def ResNet_ConvolutionBlock(Input_X,f,filters,stage,block,s=2):

        conv_name_base= 'RES' + str(stage) + block + '_branch'
        bn_name_base= 'BN' + str(stage) + block + '_branch'

        F1,F2,F3=filters

        Shortcut_X=Input_X

        Input_X=Conv2D(
            filters=F1,
            kernel_size=(1,1),
            strides=(s,s),
            name=conv_name_base+'2a',
            kernel_initializer="glorot_uniform"
        )(Input_X)
        Input_X=BatchNormalization(
            axis=3,
            name=bn_name_base+'2a'
        )(Input_X)
        Input_X=Activation('relu')(Input_X)

        Input_X=Conv2D(
            filters=F2,
            kernel_size=(f,f),
            strides=(1,1),
            padding="same",
            name=conv_name_base+'2b',
            kernel_initializer="glorot_uniform"
        )(Input_X)
        Input_X=BatchNormalization(
            axis=3,
            name=bn_name_base+'2b'
        )(Input_X)
        Input_X=Activation('relu')(Input_X)

        Input_X=Conv2D(
            filters=F3,
            kernel_size=(1,1),
            strides=(1,1),
            padding="valid",
            name=conv_name_base+'2c',
            kernel_initializer="glorot_uniform"
        )(Input_X)
        Input_X=BatchNormalization(
            axis=3,
            name=bn_name_base+'2c'
        )(Input_X)

        Shortcut_X=Conv2D(
            filters=F3,
            kernel_size=(1,1),
            strides=(s,s),
            padding="valid",
            name=conv_name_base+'1',
            kernel_initializer="glorot_uniform"
        )(Shortcut_X)
        Shortcut_X=BatchNormalization(
            axis=3,
            name=bn_name_base+'1'
        )(Shortcut_X)


        Input_X=Add()([Input_X,Shortcut_X])
        Input_X=Activation('relu')(Input_X)

        return Input_X

    def ResNet50(input_shape=(32,32,3),classes=100):

        """
        Implementation of the popular ResNet50 the following architecture:

        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        """

        X_Input=Input(shape=input_shape)

        Input_X=ZeroPadding2D((3,3))(X_Input)

        #Stage-1
        #7*7*64
        Input_X=Conv2D(
            filters=64,
            kernel_size=(7,7),
            strides=(2,2),
            name='conv1',
            kernel_initializer="glorot_uniform"
        )(Input_X)
        Input_X=BatchNormalization(
            axis=3,
            name='bn_conv1'
        )(Input_X)
        Input_X=Activation('relu')(Input_X)
        Input_X=MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2)
        )(Input_X)

        #Stage-2
        Input_X=ResNet50.ResNet_ConvolutionBlock(
            Input_X=Input_X,
            f=3,
            filters=[64,64,256],
            stage=2,
            block='a',
            s=1
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[64,64,256],
            stage=2,
            block='b',
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[64,64,256],
            stage=2,
            block='c',
        )

        #stage-3
        Input_X=ResNet50.ResNet_ConvolutionBlock(
            Input_X=Input_X,
            f=3,
            filters=[128,128,512],
            stage=3,
            block='a',
            s=2
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[128,128,512],
            stage=3,
            block='b',
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[128,128,512],
            stage=3,
            block='c',
        )
        Input_X=ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[128,128,512],
            stage=3,
            block='d',
        )

        #stage-4
        Input_X=ResNet50.ResNet_ConvolutionBlock(
            Input_X=Input_X,
            f=3,
            filters=[256,256,1024],
            stage=4,
            block='a',
            s=2
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[256,256,1024],
            stage=4,
            block='b',
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[256,256,1024],
            stage=4,
            block='c',
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[256,256,1024],
            stage=4,
            block='d',
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[256,256,1024],
            stage=4,
            block='e',
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[256,256,1024],
            stage=4,
            block='f',
        )

        #stage-5
        Input_X=ResNet50.ResNet_ConvolutionBlock(
            Input_X=Input_X,
            f=3,
            filters=[512,512,2048],
            stage=5,
            block='a',
            s=2
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[512,512,2048],
            stage=5,
            block='b',
        )
        Input_X=ResNet50.ResNet_IdentityBlock(
            Input_X=Input_X,
            f=3,
            filters=[512,512,2048],
            stage=5,
            block='c',
        )

        Input_X=AveragePooling2D(
            pool_size=(2,2),
            padding='same'
        )(Input_X)

        Input_X=Flatten()(Input_X)
        Input_X=Dense(
            classes,
            activation='softmax',
            name='fc'+str(classes),
            kernel_initializer="glorot_uniform"
        )(Input_X)

        model=Model(inputs=X_Input,outputs=Input_X,name='ResNet50')

        return model


    



