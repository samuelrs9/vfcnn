import os
import sys
from pkg_resources import split_sections
local_path = os.path.dirname(__file__)
if local_path not in sys.path:
    sys.path.append(local_path)
    
import tensorflow as tf
from tensorflow.keras.layers import *
from custom_layers import Normalize,ArgMax,Split,SampleSDF

class VoxelizedFluidCNN:

    def __init__(self,tasks,input_shape=None):
        self.available_tasks = ['boundary','normal','sdf']
        self.tasks = tasks

        self.split_input = Split(splits=2,axis=-1,name='split_input')
        self.concat_normal = Concatenate(axis=-1,name='concat_normal')
        self.concat_sdf = Concatenate(axis=-1,name='concat_sdf')
        self.arg_max = ArgMax(name='boundary_label')
        self.multiply_boundary = Multiply(name='multiply_boundary')
        self.multiply_target = Multiply(name='multiply_target')

        self.backbone_block = self.get_backbone_block()
        self.boundary_block = self.get_boundary_block()
        self.normal_block = self.get_normal_block()        

        self.keras_model = self.build_model(input_shape)

    def build_model(self,input_shape):
        inputs = Input(input_shape)
        
        input0,input1 = self.split_input(inputs)
        
        # Backbone block
        backbone_output = self.call_block(input0,self.backbone_block)
        outputs = []
        
        # Boundary block
        if 'boundary' in self.tasks:
            boundary_output = self.call_block(backbone_output,self.boundary_block)
            outputs.append(boundary_output)
        
        # Normal block
        if 'normal' in self.tasks:
            normal_output = self.call_block(backbone_output,self.normal_block)
            boundary_labels = self.arg_max(boundary_output)
            normal_output = self.multiply_boundary([normal_output,boundary_labels])
            outputs.append(normal_output)
        
        # SDF block
        if 'sdf' in self.tasks:
            pass

        outputs = self.concatenate(outputs)
        outputs = self.multiply_target([outputs,input1])

        return tf.keras.Model(inputs,outputs)

    def call_block(self,x,layer_block):
        for layer in layer_block:
            x = layer(x)
        return x

    def get_backbone_block(self):
        """ Sequência de camadas convolucionais para extração de features. """
        backbone_block = [
            Conv3D(4,(3,3,3), activation='relu',padding="same"),
            Conv3D(4,(3,3,3), activation='relu',padding="same"),
            Conv3D(4,(3,3,3), activation='relu',padding="same"),
            Conv3D(4,(3,3,3), activation='relu',padding="same"),
            MaxPooling3D(pool_size=(2,2,2)),
            BatchNormalization(momentum=0.9),
            Conv3D(8,(3,3,3), activation='relu',padding="same"),
            Conv3D(8,(3,3,3), activation='relu',padding="same"),
            MaxPooling3D(pool_size=(2,2,2)),
            BatchNormalization(momentum=0.9),
            Conv3D(16,(3,3,3), activation='relu',padding="same"),
            MaxPooling3D(pool_size=(2,2,2)),
            BatchNormalization(momentum=0.9)]
        return backbone_block
    
    def get_boundary_block(self):
        """ Sequência de camadas deconvolucionais para previsão dos labels. """
        boundary_block = [
            Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='relu'),
            BatchNormalization(momentum=0.9),
            Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='relu'),
            BatchNormalization(momentum=0.9),
            Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='relu'),
            BatchNormalization(momentum=0.9),
            Conv3D(2,(1,1,1),activation='softmax',padding='same')]
        return boundary_block

    def get_normal_block(self):
        """ Sequência de camadas deconvolucionais para previsão das normais. """
        normal_block = [       
            Conv3DTranspose(16,(3,3,3),strides=(2,2,2),activation='linear'),
            BatchNormalization(momentum=0.9),
            Conv3DTranspose(8,(3,3,3),strides=(2,2,2),activation='linear'),
            BatchNormalization(momentum=0.9),
            Conv3DTranspose(4,(3,3,3),strides=(2,2,2),activation='linear'),
            BatchNormalization(momentum=0.9),
            Conv3D(3,(1,1,1),activation='linear',padding='same'),
            Normalize(threshold=0.1,name='normalize')]
        return normal_block

    def get_sdf_block(sefl):
        """ Sequência de camadas convolutionais para estimativa da SDF. """
        sdf_block = [      
            SampleSDF(shitft=10,axis_direction=[1,-1]),
            Conv3D(1,(1,1,1),activation='linear',padding='same')]
        return sdf_block

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
        input_shape=(31,31,2),num_classes=2,pre_trained_model=None,
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