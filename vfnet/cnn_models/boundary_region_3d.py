import os
import tensorflow as tf
from tensorflow.keras.layers import *

class Models31:
    """
    Modelos de CNN3D para a abordagem 2 com input shape (31,31,31,2).
    Última modificação: 24/02/2022.
    
    Common Kwargs:
        input_shape:
        num_classes:
        pre_trained_model:
        transfer_weights:
        trainable_weigths:

    """
    @staticmethod
    def model_31_9_C4C4M2_C8C8M2_C16C16M2_C32C32M2_CT16_CT8_CT4_CT2_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
        # Sequência de camadas convolucionais e de pooling 
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(input_nn)
        x = Conv3D(4,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)

        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(8,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)

        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(16,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)

        x = Conv3D(32,(3,3,3), activation='relu',padding="same")(x)
        x = Conv3D(32,(3,3,3), activation='relu',padding="same")(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        
        # Saída
        outputs = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(outputs,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4M2_C8C8M2_C16C16M2_C32C32M2_CT16_CT8_CT4_CT2_C1')
            
        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:12]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_9_C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        outputs = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(outputs,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:16]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_10_C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        outputs = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(outputs,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1')        

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:16]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_10_C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        outputs = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(outputs,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1')        

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:16]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_11_C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        outputs = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(outputs,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1')        

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:16]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_12_C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        outputs = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(outputs,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1')        

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:16]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_13_C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Saída
        outputs = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(outputs,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4M2B_C8C8M2B_C16C16M2B_C32C32M2B_CT16B_CT8B_CT4B_CT2B_C1')        

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:16]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_9_C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_11_C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(2,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT8B_CT4B_CT2B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_9_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_11_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT16B_CT8B_CT4B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model

    @staticmethod
    def model_31_9_C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model


    @staticmethod
    def model_31_10_C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model


    @staticmethod
    def model_31_11_C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model


    @staticmethod
    def model_31_12_C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model


    @staticmethod
    def model_31_13_C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1(**kwargs):
        # Entrada
        inputs = Input(shape=kwargs['input_shape'])
        input_nn,input_target = tf.split(inputs,num_or_size_splits=2,axis=4)
                
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

        # Sequência de camadas deconvolucionais
        #x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        #x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(32,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # 2: k=(3,3) e stride=(2,2) -> 5
        # 5: k=(3,3) e stride=(2,2) -> 11
        # 11: k=(2,2) e stride=(2,2) -> 22
        # 22: k=(3,3) e stride=(2,2) -> 45
        
        # Saída
        x = Conv3D(kwargs['num_classes'],(1,1,1),activation='softmax',padding='same')(x)
        outputs = tf.multiply(x,input_target)
            
        model = tf.keras.Model(inputs,outputs,name='C4C4C4C4M2B_C8C8M2B_C16M2B_CT32B_CT16B_CT8B_C1')

        # Tranferência de pesos do modelo de classificação pré-treinado
        if kwargs['transfer_weights']:        
            for i,layer in enumerate(kwargs['pre_trained_model'].layers[0:13]):
                weights = layer.get_weights()
                model.layers[i+2].set_weights(weights)
                model.layers[i+2].trainable = kwargs['trainable_weigths']
        
        model.summary()
        return model
