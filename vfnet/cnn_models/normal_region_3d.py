import os
import sys
local_path = os.path.dirname(__file__)
if local_path not in sys.path:
    sys.path.append(local_path)
    
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from custom_layers import Normalize,ArgMax

class Models25:
    """
    Modelos de CNN 3D para a abordagem regional com input shape (25,25,25,3).
    Última modificação: 27/05/2022.
    
    Common Args:
        num_classes:
        input_shape:
            
    """
    @staticmethod
    def model_25_C1CC1C1M2B_C2C2M2B_C4M2B_CT4B_CT2B_CT2B_C3_LN(
        input_shape=(25,25,25,2),num_classes=2,pre_trained_model=None,
        transfer_weights=False,trainable_weigths=False):
        # Entrada
        inputs = Input(shape=input_shape)
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=-1)
                
        # Sequência de camadas convolucionais e de pooling 
        x = Conv3D(1,(3,3,3), activation='relu',padding="same")(input_nn)
        x = Conv3D(1,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(1,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(1,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(2,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(2,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Sequência de camadas deconvolucionais para previsão dos labels
        labels = Conv3DTranspose(4,(2,2,2),strides=(2,2,2),activation='relu')(x)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(2,(2,2,2),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3D(num_classes,(1,1,1),activation='softmax',padding='same')(labels)        
        argmax = ArgMax()(labels)

        # Sequência de camadas deconvolucionais para previsão das normais
        normal = Conv3DTranspose(4,(2,2,2),strides=(2,2,2),activation='linear')(x)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(2,(2,2,2),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3D(3,(1,1,1),activation='linear',padding='same')(normal)
        normal = Normalize(threshold=0.1,name='normalize')(normal)
        normal = tf.multiply(normal,argmax)

        output = tf.concat([labels,normal],axis=-1)
        output = tf.multiply(output,input_target)
        
        model = tf.keras.Model(inputs,output,name='C1CC1C1M2B_C2C2M2B_C4M2B_CT4B_CT2B_CT2B_C3_LN')

        model.summary()
        return model

    @staticmethod
    def model_25_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN(
        input_shape=(25,25,25,2),num_classes=2,pre_trained_model=None,
        transfer_weights=False,trainable_weigths=False):
        # Entrada
        inputs = Input(shape=input_shape)
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=-1)
                
        # Sequência de camadas convolucionais e de pooling 
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(input_nn)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Sequência de camadas deconvolucionais para previsão dos labels
        labels = Conv3DTranspose(16,(2,2,2),strides=(2,2,2),activation='relu')(x)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(8,(2,2,2),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3D(num_classes,(1,1,1),activation='softmax',padding='same')(labels)        
        argmax = ArgMax()(labels)

        # Sequência de camadas deconvolucionais para previsão das normais
        normal = Conv3DTranspose(16,(2,2,2),strides=(2,2,2),activation='linear')(x)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(8,(2,2,2),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3D(3,(1,1,1),activation='linear',padding='same')(normal)
        normal = Normalize(threshold=0.1,name='normalize')(normal)
        normal = tf.multiply(normal,argmax)

        output = tf.concat([labels,normal],axis=-1)
        output = tf.multiply(output,input_target)
        
        model = tf.keras.Model(inputs,output,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN')

        model.summary()
        return model


class Models31:
    """
    Modelos de CNN 3D para a abordagem regional com input shape (31,31,31,2).
    Última modificação: 11/04/2022.
    
    Common Args:
        num_classes:
        input_shape:
            
    """
    @staticmethod
    def model_31_C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2_C1_LN(
        input_shape=(31,31,2),num_classes=2,pre_trained_model=None,
        transfer_weights=False,trainable_weigths=False):
        # Entrada
        inputs = Input(shape=input_shape)
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=-1)
                
        # Sequência de camadas convolucionais e de pooling 
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(input_nn)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(32,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(32,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Sequência de camadas deconvolucionais para previsão dos labels
        labels = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3D(num_classes,(1,1,1),activation='softmax',padding='same')(labels)
        argmax = ArgMax()(labels)

        # Sequência de camadas deconvolucionais para previsão das normais
        normal = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='linear')(x)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)        
        normal = Conv3D(3,(1,1,1),activation='linear',padding='same')(normal)
        normal = Normalize(threshold=0.1,name='normalize')(normal)        
        normal = tf.multiply(normal,argmax)

        output = tf.concat([labels,normal],axis=-1)
        output = tf.multiply(output,input_target)

        model = tf.keras.Model(inputs,output,name='C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2_C1_LN')

        model.summary()
        return model

    @staticmethod
    def model_31_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN(
        input_shape=(31,31,31,2),num_classes=2,pre_trained_model=None,
        transfer_weights=False,trainable_weigths=False):
        # Entrada
        inputs = Input(shape=input_shape)
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=-1)
                
        # Sequência de camadas convolucionais e de pooling 
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(input_nn)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Sequência de camadas deconvolucionais para previsão dos labels
        labels = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3D(num_classes,(1,1,1),activation='softmax',padding='same')(labels)        
        argmax = ArgMax()(labels)

        # Sequência de camadas deconvolucionais para previsão das normais
        normal = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='linear')(x)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3D(3,(1,1,1),activation='linear',padding='same')(normal)
        normal = Normalize(threshold=0.1,name='normalize')(normal)
        normal = tf.multiply(normal,argmax)

        output = tf.concat([labels,normal],axis=-1)
        output = tf.multiply(output,input_target)
        
        model = tf.keras.Model(inputs,output,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C3_LN')

        model.summary()
        return model
        
    @staticmethod
    def model_31_C8C8C8C8M2B_C16C16M2B_C32M2B_CT32B_CT16B_CT8B_C3_LN(
        input_shape=(31,31,31,2),num_classes=2,pre_trained_model=None,
        transfer_weights=False,trainable_weigths=False):
        # Entrada
        inputs = Input(shape=input_shape)
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=-1)
                
        # Sequência de camadas convolucionais e de pooling 
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(input_nn)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(32,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Sequência de camadas deconvolucionais para previsão dos labels
        labels = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='relu')(x)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(labels)
        labels = BatchNormalization(momentum=0.9)(labels)
        labels = Conv3D(num_classes,(1,1,1),activation='softmax',padding='same')(labels)        
        argmax = ArgMax()(labels)

        # Sequência de camadas deconvolucionais para previsão das normais
        normal = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='linear')(x)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='linear')(normal)
        normal = BatchNormalization(momentum=0.9)(normal)
        normal = Conv3D(3,(1,1,1),activation='linear',padding='same')(normal)
        normal = Normalize(threshold=0.1,name='normalize')(normal)
        normal = tf.multiply(normal,argmax)

        output = tf.concat([labels,normal],axis=-1)
        output = tf.multiply(output,input_target)
        
        model = tf.keras.Model(inputs,output,name='C8C8C8C8M2B_C16C16M2B_C32M2B_CT32B_CT16B_CT8B_C3_LN')

        model.summary()
        return model        

    @staticmethod
    def model_31_C8C8C8C8M2B_C16C16M2B_C32M2B_CT32B_CT16B_CT8B_L2N3(
        input_shape=(31,31,31,2),num_classes=2,pre_trained_model=None,
        transfer_weights=False,trainable_weigths=False):
        # Entrada
        inputs = Input(shape=input_shape)
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=-1)
                
        # Sequência de camadas convolucionais e de pooling 
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(input_nn)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        x = Conv3D(32,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        labels = Conv3D(num_classes,(1,1,1),activation='softmax',padding='same')(x)

        normal = Conv3D(3,(1,1,1),activation='linear',padding='same')(x)
        normal = Normalize(threshold=0.1,name='normalize')(normal)        
        normal = tf.multiply(normal,ArgMax()(labels))

        output = tf.concat([labels,normal],axis=-1)
        output = tf.multiply(output,input_target)
        
        model = tf.keras.Model(inputs,output,name='C8C8C8C8M2B_C16C16M2B_C32M2B_CT32B_CT16B_CT8B_L2N3')

        model.summary()
        return model                