import tensorflow as tf

class MatthewsCoefficient(tf.keras.metrics.Metric):
    
    def __init__(self, name='mcc', **kwargs):
        super(MatthewsCoefficient, self).__init__(name=name, **kwargs)    
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.mcc = self.add_weight(name='mcc', initializer='zeros')
      
    def update_state(self,labels_true,labels_pred,sample_weight=None):
        batch_tp = tf.keras.metrics.TruePositives()
        batch_tn = tf.keras.metrics.TrueNegatives()
        batch_fp = tf.keras.metrics.FalsePositives()
        batch_fn = tf.keras.metrics.FalseNegatives()    
      
        batch_tp.update_state(labels_true,labels_pred)
        batch_tn.update_state(labels_true,labels_pred)
        batch_fp.update_state(labels_true,labels_pred)
        batch_fn.update_state(labels_true,labels_pred)
      
        self.tp.assign_add(float(batch_tp.result()))
        self.tn.assign_add(float(batch_tn.result()))
        self.fp.assign_add(float(batch_fp.result()))
        self.fn.assign_add(float(batch_fn.result()))
        
        tp = float(self.tp)
        tn = float(self.tn)
        fp = float(self.fp)
        fn = float(self.fn)    
      
        den = (tp * tn) - (fp * fn)
        div = ((tp + fp) * (tp + fn) * (tn + fp ) * (tn + fn))**(1/2)
        if tf.math.abs(div) > tf.keras.backend.epsilon(): 
            mcc =  den / div 
        else:
            mcc = 0.0
      
        self.mcc.assign(float(mcc))
    
    def result(self):
        return self.mcc
    
    def reset_state(self):
        self.mcc.assign(0)

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.f1score = self.add_weight(name='f1score', initializer='zeros')
      
    def update_state(self,labels_true,labels_pred,sample_weight=None):
        labels_true = tf.cast(labels_true,dtype=tf.float32)
        labels_pred = tf.cast(labels_pred,dtype=tf.float32)        
        # Precision
        batch_precision = tf.keras.metrics.Precision()
        batch_precision.update_state(labels_true,labels_pred)
        # Recall
        batch_recall = tf.keras.metrics.Recall()
        batch_recall.update_state(labels_true,labels_pred)
        
        self.precision.assign_add(float(batch_precision.result()))
        self.recall.assign_add(float(batch_recall.result()))          
        f1score = 2 * (self.precision * self.recall) / (self.precision + self.recall)      

        self.f1score.assign(float(f1score))
    
    def result(self):
        return self.f1score
    
    def reset_state(self):
        self.f1score.assign(0)

class MaskMetric(tf.keras.metrics.Metric):

    def __init__(self, name='mask_metrics', **kwargs):
        super(MaskMetric, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')        
    
    def update_state(self, mask_true, mask_pred, sample_weight=None):
        mask_true = tf.cast(mask_true,dtype=tf.float32)
        mask_pred = tf.cast(mask_pred,dtype=tf.float32)
        # Extrai os labels importantes das máscaras de labels
        labels_true,labels_pred = self.extract_target_labels(mask_true,mask_pred)
        # True positive
        batch_tp = tf.keras.metrics.TruePositives()
        batch_tp.update_state(labels_true,labels_pred)
        self.tp.assign_add(float(batch_tp.result()))
        # True negative
        batch_tn = tf.keras.metrics.TrueNegatives()
        batch_tn.update_state(labels_true,labels_pred)
        self.tn.assign_add(float(batch_tn.result()))
        # False positive
        batch_fp = tf.keras.metrics.FalsePositives()
        batch_fp.update_state(labels_true,labels_pred)
        self.fp.assign_add(float(batch_fp.result()))
        # False negative
        batch_fn = tf.keras.metrics.FalseNegatives()
        batch_fn.update_state(labels_true,labels_pred)
        self.fn.assign_add(float(batch_fn.result()))   

    def extract_target_labels(self,mask_true,mask_pred):
        # Extrai os pixels relevantes de labels_mask_true
        mask_true_reshaped = tf.reshape(mask_true,[-1,mask_true.shape[-1]])
        target_pixels = tf.math.abs(tf.reduce_sum(mask_true_reshaped,axis=-1)) > tf.keras.backend.epsilon()
        # labels true
        target_mask_true_reshaped = tf.boolean_mask(mask_true_reshaped,target_pixels,axis=0)
        labels_true = tf.argmax(target_mask_true_reshaped,axis=-1)        
        # labels pred
        mask_pred_reshaped = tf.reshape(mask_pred,[-1,mask_pred.shape[-1]])
        target_mask_pred_reshaped = tf.boolean_mask(mask_pred_reshaped,target_pixels,axis=0)
        labels_pred = tf.argmax(target_mask_pred_reshaped,axis=-1)        
        return labels_true,labels_pred

class PrecisionMask(MaskMetric):

    def __init__(self, name='precision', **kwargs):
        super(PrecisionMask, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(name='precision', initializer='zeros')
      
    def update_state(self,mask_true,mask_pred,sample_weight=None):
        super().update_state(mask_true,mask_pred)
        precision = float(self.tp) / (float(self.tp) + float(self.fp))
        self.precision.assign(precision)
    
    def result(self):
        return self.precision
    
    def reset_state(self):
        self.precision.assign(0)

class RecallMask(MaskMetric):

    def __init__(self, name='recall', **kwargs):
        super(RecallMask, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name='recall', initializer='zeros')
      
    def update_state(self,mask_true,mask_pred,sample_weight=None):
        super().update_state(mask_true,mask_pred)
        recall = float(self.tp) / (float(self.tp) + float(self.fn))
        self.recall.assign(recall)
    
    def result(self):
        return self.recall
    
    def reset_state(self):
        self.recall.assign(0)

class F1ScoreMask(MaskMetric):

    def __init__(self, name='f1score', **kwargs):
        super(F1ScoreMask, self).__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1score', initializer='zeros')
      
    def update_state(self,mask_true,mask_pred,sample_weight=None):
        super().update_state(mask_true,mask_pred)
        precision = float(self.tp) / (float(self.tp) + float(self.fp))
        recall = float(self.tp) / (float(self.tp) + float(self.fn))
        f1 = 2*(precision*recall)/(precision+recall)
        self.f1.assign(f1)
    
    def result(self):
        return self.f1
    
    def reset_state(self):
        self.f1.assign(0)

class MatthewsCoefficientMask(MaskMetric):
    """" 
    Matthews correlation coefficient for regionwise approach. 
    """
    def __init__(self, name='mcc', **kwargs):
        super(MatthewsCoefficientMask, self).__init__(name=name, **kwargs)    
        self.mcc = self.add_weight(name='mcc', initializer='zeros')
      
    def update_state(self, mask_true, mask_pred, sample_weight=None):
        super().update_state(mask_true,mask_pred)
        tp = float(self.tp)
        tn = float(self.tn)
        fp = float(self.fp)
        fn = float(self.fn)          
        den = (tp * tn) - (fp * fn)
        div = ((tp + fp) * (tp + fn) * (tn + fp ) * (tn + fn))**(1/2)
        if tf.math.abs(div) > tf.keras.backend.epsilon(): 
            mcc =  den / div 
        else:
            mcc = 0.0      
        self.mcc.assign(float(mcc))
    
    def result(self):
        return self.mcc
    
    def reset_state(self):
        self.mcc.assign(0)
 
class MatthewsCoefficientApproach2(tf.keras.metrics.Metric):
    """" 
    Matthews correlation coefficient for regionwise approach. 
    """
    def __init__(self, name='mcc', **kwargs):
        super(MatthewsCoefficientApproach2, self).__init__(name=name, **kwargs)    
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.mcc = self.add_weight(name='mcc', initializer='zeros')
      
    def update_state(self, mask_true, mask_pred, sample_weight=None):
        mask_true = tf.cast(mask_true,dtype=tf.float32)
        mask_pred = tf.cast(mask_pred,dtype=tf.float32)        
        # Transforma as imagens de output em um único vetor de labels
        pixel_vector_true = tf.reshape(mask_true,[-1,mask_true.shape[-1]])
        no_null_true = tf.math.abs(tf.reduce_sum(pixel_vector_true,axis=-1)) > tf.keras.backend.epsilon()
        pixel_vector_true_no_null = tf.boolean_mask(pixel_vector_true,no_null_true,axis=0)
        labels_true = tf.argmax(pixel_vector_true_no_null,axis=1)
      
        # Transforma as imagens preditas em um único vetor de labels
        pixel_vector_pred = tf.reshape(mask_pred,[-1,2])
        no_null_pred = tf.math.abs(tf.reduce_sum(pixel_vector_pred,axis=-1)) > tf.keras.backend.epsilon()
        pixel_vector_pred_no_null = tf.boolean_mask(pixel_vector_pred,no_null_pred,axis=0)    
        labels_pred = tf.argmax(pixel_vector_pred_no_null,axis=1)
      
        batch_tp = tf.keras.metrics.TruePositives()
        batch_tn = tf.keras.metrics.TrueNegatives()
        batch_fp = tf.keras.metrics.FalsePositives()
        batch_fn = tf.keras.metrics.FalseNegatives()    
      
        batch_tp.update_state(labels_true,labels_pred)
        batch_tn.update_state(labels_true,labels_pred)
        batch_fp.update_state(labels_true,labels_pred)
        batch_fn.update_state(labels_true,labels_pred)
      
        self.tp.assign_add(float(batch_tp.result()))
        self.tn.assign_add(float(batch_tn.result()))
        self.fp.assign_add(float(batch_fp.result()))
        self.fn.assign_add(float(batch_fn.result()))
        
        tp = float(self.tp)
        tn = float(self.tn)
        fp = float(self.fp)
        fn = float(self.fn)
      
        den = (tp * tn) - (fp * fn)
        div = ((tp + fp) * (tp + fn) * (tn + fp ) * (tn + fn))**(1/2)
        if tf.math.abs(div) > tf.keras.backend.epsilon(): 
            mcc =  den / div 
        else:
            mcc = 0.0
      
        self.mcc.assign(float(mcc))
    
    def result(self):
        return self.mcc
    
    def reset_state(self):
        self.mcc.assign(0)
