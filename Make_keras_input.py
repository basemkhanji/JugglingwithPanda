import pandas as pd
import numpy as np
import swifter
from collections import Counter
import statistics
from statistics import mode
import pickle
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from tensorflow import keras
from keras import backend as K
from keras.models import Model
import keras.layers
from numpy import savez_compressed
from sklearn.preprocessing import MinMaxScaler

# Add the labels to the data frame  
df = pd.read_pickle('./Data/MC/Bd2JpsiKst_MC.pkl')
print(df.columns)
# turn B_ID into int labels                                                                                                                                   
# only for MC                                                                                                                                                 
df.loc[df['B_TRUEID']> 0 ,  'label' ] = 1
df.loc[df['B_TRUEID']< 0 ,  'label' ] = 0

#drop non-used columns                                                                                                                                       
drop_col = ['B_FDCHI2_OWNPV','Tr_MC_Bsignal_OSCIL','B_TRUEID','Tr_T_MinIPChi2', 'Tr_T_AALLSAMEBPV','Tr_T_Best_PAIR_DCHI2']                                   
df = df.drop(drop_col   , axis=1)                           
# define the track features which are going to be used in Keras model :
track_features= [
    'Tr_T_PROBNNe', 'Tr_T_PROBNNk', 'Tr_T_PROBNNmu','Tr_T_PROBNNp','Tr_T_PROBNNpi','Tr_T_PROBNNghost',
    'Tr_T_VeloCharge','Tr_T_P','Tr_T_PT','Tr_T_E','Tr_T_Charge',
    'Tr_T_Phi', 'Tr_T_Eta', 'Tr_T_BPVIP', 'Tr_T_BPVIPCHI2','Tr_T_SumBDT_ult','Tr_T_IP_trMother',
    'Tr_T_TrFIRSTHITZ','Tr_T_diff_z', 'Tr_T_cos_diff_phi' , 'Tr_T_diff_eta','Tr_T_P_proj'
]

df_input = df[track_features]
# track features do not have the same length, this is not allowed in keras(notice that this is possible in pytorch!)                                                                                   
# we use padding                                                                                                                                              
from keras.preprocessing.sequence import pad_sequences
for col in track_features:
    print(col)
    df_input[col] = pad_sequences(df_input[col], maxlen= 40,padding='post',value= -999 ,dtype='float32').tolist()
    df_input[col] = pd.Series(df_input[col]).apply(np.asarray)
    
# For keras we need to feed a flat array: 
# the follwoing trick flatten all columns in each row (this does not suite TimeDistributed layer):                                                             
#df_input_np = df_input.to_numpy()                                                                                                                            
#df_input_np = np.asarray(list( np.concatenate(x).ravel() for x in df_input_np))                                                                              
# the actuall solution be be adopted here , however, is : to convert the array into list:                                                                
df_input_np = df_input.to_numpy()
df_input_np = np.array([x.tolist() for x in df_input_np])
# this is suitable for Keras TimeDistributed layer):
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( df_input_np  , df['label'].values , test_size=0.10, random_state=42)
axtracks      = 40
epochs         = 50
batch_size     = 2 ** 12
trackShape     = (len(df_input[track_features].columns),40 ) # nTracks, nFeatures                                                                             
nB = 2
# the follwoing model is adopted from Dan's O'Hanlon's model (inclusive tagging at LHCb)
trackInput = keras.layers.Input(trackShape)
tracks = keras.layers.Masking(mask_value = -999, name = 'mask')(trackInput)
tracks = keras.layers.TimeDistributed(keras.layers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)
tracks = keras.layers.TimeDistributed(keras.layers.Dense(32, activation = 'relu'), name = 'td_dense2')(tracks)
tracks = keras.layers.GRU(32, activation = 'relu', return_sequences = True, name = 'track_gru')(tracks)
tracks = keras.layers.GRU(32, activation = 'relu', return_sequences = False, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)
# Try also concatenating to the time axis?                                                                                                                    
#tracks = keras.layers.Concatenate(-1)([tracks, bType])                                                                                                       
tracks = keras.layers.Dense(32, activation = 'relu', name = 'out_dense_1')(tracks)
tracks = keras.layers.Dense(32, activation = 'relu', name = 'out_dense_2')(tracks)
tracks = keras.layers.Dense(32, activation = 'relu', name = 'out_dense_3')(tracks)
outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)
model = Model(inputs = trackInput , outputs = outputTag)
#model = Model(inputs = [trackInput, bTypeInput], outputs = outputTag)                                                                                        
model.summary()
adam = Adam(lr = 0.001, amsgrad = True)
earlyStopping = EarlyStopping(patience = 100)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

history =model.fit(X_train , y_train,
                   #callbacks = [earlyStopping],                                                                                                              
                   epochs = 100, verbose = 1)

from sklearn.metrics import roc_curve

y_pred_keras = model.predict(X_test  ).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
print(roc_auc_score( y_test, y_pred_keras ) )

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
    


