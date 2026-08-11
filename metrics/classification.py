import numpy as np
from sklearn.metrics import confusion_matrix

class Report:
           
        def __init__(self,*args,labels=None):
            """
            Construtor.
            Última modificação: 19/05/2021.
            
            Args:
                *args:                
            """
            if len(args)==1:
                self.confusion_matrix = np.array(args[0])
            elif len(args)==2:
                self.true,self.pred = args[0],args[1]
                self.confusion_matrix = confusion_matrix(self.true,self.pred,labels=labels)
            else: 
                print('Incorrect arguments!')
                return
            
            self.FP = (self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)).astype(float)
            self.FN = (self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)).astype(float)
            self.TP = (np.diag(self.confusion_matrix)).astype(float)
            self.TN = (self.confusion_matrix.sum() - (self.FP + self.FN + self.TP)).astype(float)
            
        def recall(self,positive=1):
            """ 
            Sensitivity, hit rate, recall, or true positive rate.
            Última modificação: 23/09/2021.
            
            Args: 
                positive: classe positiva.
            """             
            return (self.TP/(self.TP + self.FN))[positive]
        
        def precision(self,positive=1):
            """ 
            Precision or positive predictive value (PPV).
            Última modificação: 23/09/2021.
            
            Args: 
                positive: classe positiva.
            """ 
            return (self.TP/(self.TP + self.FP))[positive]
            
        def tnr(self,positive=1):
            """ 
            Specificity or true negative rate.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """             
            return (self.TN/(self.TN + self.FP))[positive]   
             
        def npv(self,positive=1):
            """ 
            Negative predictive value.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """
            return (self.TN/(self.TN + self.FN))[positive]
               
        def fpr(self,positive=1):
            """ 
            Fall out or false positive rate.
            Última modificação: 19/05/2021
            
            Args: 
                positive: classe positiva.
            """  
            return (self.FP/(self.FP + self.TN))[positive]

        def fnr(self,positive=1):
            """ 
            False negative rate.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """              
            return (self.FN/(self.TP + self.FN))[positive]

        def fdr(self,positive=1):
            """ 
            False discovery rate.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """              
            return (self.FP/(self.TP + self.FP))[positive]

        def overall_accuracy(self,positive=1):
            """ 
            Overall accuracy.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """                     
            return ((self.TP+self.TN)/(self.TP+self.FP+self.FN+self.TN))[positive]
        
        def average_false_rate(self,positive=1):
            """ 
            Average false rate.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """                                      
            return 0.5*(self.fpr(positive) + self.fnr(positive))
                       
        def combined_metric(self,positive=1):
            """ 
            Combined metric.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """                                
            return self.recall(positive)*(1 - self.fpr(positive))

        def f1_score(self,positive=1):
            """ 
            F1 Score.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """
            return 2*self.precision(positive)*self.recall(positive)\
                    /(self.precision(positive)+self.recall(positive))

        def matthews_coefficient(self,positive=1):
            """ 
            Matthews correlation coefficient.
            Última modificação: 19/05/2021.
            
            Args: 
                positive: classe positiva.
            """            
            return ((self.TP*self.TN-self.FP*self.FN)/np.sqrt((self.TP+self.FP)\
                    *(self.TP + self.FN)*(self.TN + self.FP)*(self.TN + self.FN)))[positive]



