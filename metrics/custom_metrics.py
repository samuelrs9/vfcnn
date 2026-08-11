import tensorflow as tf

class PointwiseMetric(tf.keras.metrics.Metric):

    def __init__(self,slice=None,name='point_wise_metric',dtype=None,**kwargs):
        super(PointwiseMetric,self).__init__(name,dtype,**kwargs)
        self.slice = slice
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')        
    
    def update_state(self,y_true,y_pred,sample_weight=None):
        y_true,y_pred = self.cast_to_tensor(y_true,y_pred)
        y_true,y_pred = self.slice_targets(y_true,y_pred)
        batch_tp = tf.keras.metrics.TruePositives()
        batch_tn = tf.keras.metrics.TrueNegatives()
        batch_fp = tf.keras.metrics.FalsePositives()
        batch_fn = tf.keras.metrics.FalseNegatives()
      
        batch_tp.update_state(y_true,y_pred)
        batch_tn.update_state(y_true,y_pred)
        batch_fp.update_state(y_true,y_pred)
        batch_fn.update_state(y_true,y_pred)
      
        self.tp.assign_add(float(batch_tp.result()))
        self.tn.assign_add(float(batch_tn.result()))
        self.fp.assign_add(float(batch_fp.result()))

        self.fn.assign_add(float(batch_fn.result()))

    def slice_targets(self,y_true,y_pred,sample_weight=None):
        if self.slice is not None:
            y_true = tf.slice(
                y_true,self.slice['begin'],self.slice['size'])
            y_pred = tf.slice(
                y_pred,self.slice['begin'],self.slice['size'])
        return y_true,y_pred

    def cast_to_tensor(self,y_true,y_pred):
        y_true = tf.cast(y_true,dtype=tf.float32)
        y_pred = tf.cast(y_pred,dtype=tf.float32)
        return y_true,y_pred

    def result(self):
        return tf.constant(
            [[self.tp.numpy(),self.tn.numpy()],
            [self.fn.numpy(),self.fp.numpy()]])

class PrecisionPW(PointwiseMetric):

    def __init__(self,slice=None,name='precision',**kwargs):
        super(PrecisionPW,self).__init__(slice=slice,name=name,**kwargs)
        self.precision = self.add_weight(name='precision',initializer='zeros')
      
    def update_state(self,y_true,y_pred,sample_weight=None):
        super().update_state(y_true,y_pred)
        precision = float(self.tp)/(float(self.tp) + float(self.fp))
        self.precision.assign(float(precision))
    
    def result(self):
        return self.precision
    
    def reset_state(self):
        self.precision.assign(0)

class RecallPW(PointwiseMetric):

    def __init__(self,slice=None,name='recall',**kwargs):
        super(RecallPW,self).__init__(slice=slice,name=name,**kwargs)
        self.recall = self.add_weight(name='recall',initializer='zeros')
      
    def update_state(self,y_true,y_pred,sample_weight=None):
        super().update_state(y_true,y_pred)
        recall = float(self.tp)/(float(self.tp) + float(self.fn))
        self.recall.assign(float(recall))
    
    def result(self):
        return self.recall
    
    def reset_state(self):
        self.recall.assign(0)        

class F1ScorePW(PointwiseMetric):

    def __init__(self,slice=None,name='f1score',**kwargs):
        super(F1ScorePW,self).__init__(slice=slice,name=name,**kwargs)
        self.f1score = self.add_weight(name='f1score',initializer='zeros')
      
    def update_state(self,y_true,y_pred,sample_weight=None):
        super().update_state(y_true,y_pred)
        precision = float(self.tp)/(float(self.tp) + float(self.fp))
        recall = float(self.tp)/(float(self.tp) + float(self.fn))    
        f1score = 2*(precision*recall)/(precision+recall)
        self.f1score.assign(float(f1score))
    
    def result(self):
        return self.f1score
    
    def reset_state(self):
        self.f1score.assign(0)

class MatthewsCoefficientPW(PointwiseMetric):
    
    def __init__(self,name='mcc',**kwargs):
        super(MatthewsCoefficientPW,self).__init__(name=name, **kwargs)
        self.mcc = self.add_weight(name='mcc',initializer='zeros')
      
    def update_state(self,y_true,y_pred,sample_weight=None):
        super().update_state(y_true,y_pred)
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

class RegionWiseMetric(tf.keras.metrics.Metric):

    def __init__(self,slice=None,name='region_wise_metric',**kwargs):
        super(RegionWiseMetric,self).__init__(name=name,**kwargs)
        self.slice = slice
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')        
    
    def update_state(self,mask_true,mask_pred,sample_weight=None):
        mask_true,mask_pred = self.cast_to_tensor(mask_true,mask_pred)
        mask_true,mask_pred = self.slice_targets(mask_true,mask_pred)
        # Extrai os labels importantes das máscaras de labels
        y_true,y_pred = self.extract_target_labels(mask_true,mask_pred)
        # True positive
        batch_tp = tf.keras.metrics.TruePositives()
        batch_tp.update_state(y_true,y_pred)
        self.tp.assign_add(float(batch_tp.result()))
        # True negative
        batch_tn = tf.keras.metrics.TrueNegatives()
        batch_tn.update_state(y_true,y_pred)
        self.tn.assign_add(float(batch_tn.result()))
        # False positive
        batch_fp = tf.keras.metrics.FalsePositives()
        batch_fp.update_state(y_true,y_pred)
        self.fp.assign_add(float(batch_fp.result()))
        # False negative
        batch_fn = tf.keras.metrics.FalseNegatives()
        batch_fn.update_state(y_true,y_pred)
        self.fn.assign_add(float(batch_fn.result()))   

    def extract_target_labels(self,mask_true,mask_pred):
        # Extrai os pixels relevantes de labels_mask_true
        mask_true_reshaped = tf.reshape(mask_true,[-1,mask_true.shape[-1]])
        target_pixels = tf.math.abs(tf.reduce_sum(mask_true_reshaped,axis=-1)) > tf.keras.backend.epsilon()
        # labels true
        target_mask_true_reshaped = tf.boolean_mask(mask_true_reshaped,target_pixels,axis=0)
        y_true = tf.argmax(target_mask_true_reshaped,axis=-1)        
        # labels pred
        mask_pred_reshaped = tf.reshape(mask_pred,[-1,mask_pred.shape[-1]])
        target_mask_pred_reshaped = tf.boolean_mask(mask_pred_reshaped,target_pixels,axis=0)
        y_pred = tf.argmax(target_mask_pred_reshaped,axis=-1)        
        return y_true,y_pred

    def slice_targets(self,mask_true,mask_pred):
        if self.slice is not None:
            mask_true = tf.slice(
                mask_true,self.slice['begin'],self.slice['size'])
            mask_pred = tf.slice(
                mask_pred,self.slice['begin'],self.slice['size'])
        return mask_true,mask_pred

    def cast_to_tensor(self,mask_true,mask_pred):
        mask_true = tf.cast(mask_true,dtype=tf.float32)
        mask_pred = tf.cast(mask_pred,dtype=tf.float32)     
        return mask_true,mask_pred

class PrecisionRW(RegionWiseMetric):

    def __init__(self,name='precision',**kwargs):
        super(PrecisionRW, self).__init__(name=name,**kwargs)
        self.precision = self.add_weight(name='precision',initializer='zeros')
      
    def update_state(self,mask_true,mask_pred,sample_weight=None):
        super().update_state(mask_true,mask_pred)
        precision = float(self.tp) / (float(self.tp) + float(self.fp))
        self.precision.assign(precision)
    
    def result(self):
        return self.precision
    
    def reset_state(self):
        self.precision.assign(0)

class RecallRW(RegionWiseMetric):

    def __init__(self, name='recall', **kwargs):
        super(RecallRW, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name='recall', initializer='zeros')
      
    def update_state(self,mask_true,mask_pred,sample_weight=None):
        super().update_state(mask_true,mask_pred)
        recall = float(self.tp) / (float(self.tp) + float(self.fn))
        self.recall.assign(recall)
    
    def result(self):
        return self.recall
    
    def reset_state(self):
        self.recall.assign(0)

class F1ScoreRW(RegionWiseMetric):

    def __init__(self, name='f1score', **kwargs):
        super(F1ScoreRW, self).__init__(name=name, **kwargs)
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

class MatthewsCoefficientRW(RegionWiseMetric):
    """" 
    Matthews correlation coefficient for regionwise approach. 
    """
    def __init__(self,name='mcc',**kwargs):
        super(MatthewsCoefficientRW, self).__init__(name=name,**kwargs)    
        self.mcc = self.add_weight(name='mcc',initializer='zeros')
      
    def update_state(self,mask_true,mask_pred,sample_weight=None):
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