import pandas as pd
import numpy as np
import swifter
import pickle
import time
import os , glob
from keras.preprocessing.sequence import pad_sequences

# Loop over ALL files in a directory based on a given pattern (defined by the user)
Data = True
suffix  = 'Data' if Data else 'MC'
os.chdir('./Data/'+ suffix  +'/')
name_pattern  = '*_'+suffix +'_fltrd_*.pkl'
# all the files are now in a list object:
files_l = [f for f in glob.glob( name_pattern )]
print(files_l)

# we want to reform the pickle file we have prepared using Preparedata.py
df_l = []
for f in files_l:
    print('===== Reading ', f , ' file ... =====' )
    df = pd.read_pickle(f)
    print(df)
    # get the names of sub-index-realted features 
    tr_l = [x for x in df.columns if x.startswith("Tr_")]
    vec_l = tr_l
    # get the names of index-related features 
    scr_l = [col for col in df.columns if col not in tr_l]
    # now do the flippty-flop through groupby combined with aggreation function in panda:
    df = df.groupby(scr_l,sort=False)[vec_l].agg(list).reset_index()
    # reset index is helpful not to keep the old index ordering which will come out random after the last procedure
    print(df)
    df_l.append(df)
    print('===== dataframe for file' , f , ' is appended =====')
    # voila the data set now is in the old shape !
df = pd.concat(df_l)
# Normalize your data                                                                                                                                         
#                                                                                                                                                             
df.to_pickle('./Bd2JpsiKst_'+ suffix +'.pkl')

