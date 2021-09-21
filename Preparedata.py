import numpy as np
from tqdm import tqdm
import pandas as pd
import uproot
from sklearn.preprocessing import MinMaxScaler


Run2_MC = '/eos/lhcb/wg/FlavourTagging/tuples/development/IFT/data_Feb2021/fttuple_data_2016_28r2_TrT.root'
tree    = 'Bd2JpsiKstarDetached/DecayTree'
# there are two indexes in this data-set: B-candidate index (main index) and dependent one called B_len, this is the track index

br_l = uproot.open(Run2_MC)[tree].keys()
# get all Features which start with "Tr_"
tr_l = [x for x in br_l if x.startswith("Tr_")]
# get all Features which start with "B_"
b_vars = [ x for x in br_l if x.startswith("B_")]

print('Number of columns: '+str(len(b_vars)))
scaler = MinMaxScaler()
events = uproot.open(Run2_MC+':'+ tree )
i = 0
# run over the data set in chnuk of 200 MB to reduce the load on the RAM, delete the df after saving it at the end of the loop:
for df in events.iterate(step_size="200 MB", expressions= b_vars, library="pd"):
    # cast to reduce the size of the output file:                                                                                                                    
    df[df.select_dtypes(bool).columns] = df.select_dtypes(bool).astype('int32')
    df[df.select_dtypes('float64').columns] = df.select_dtypes('float64').astype('float32')
    df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('int32')
    df[df.select_dtypes(object).columns] = df.select_dtypes(object).astype('float32')
    #print(df.dtypes)   
    # let us start the juggling !! : 
    # to select in basis of a given index (here we would like to select based on track feature ) we follow two step procedure:
    #1- explode everything to be dependent on the track index 
    df = df.apply( lambda x: x.explode() if x.name.startswith("Tr_") else x)
    #2- Select :
    df = df.query('(Tr_T_AALLSAMEBPV==1 | Tr_T_BPVIPCHI2>6) & Tr_T_TRGHOSTPROB<0.35 & Tr_T_P>2000 & B_DIRA_OWNPV>0.9995')
    # build additonal features :
    df['Tr_T_diff_z']       = df['B_OWNPV_Z'] -  df['Tr_T_TrFIRSTHITZ']
    df['Tr_T_cos_diff_phi'] = np.cos( df['B_LOKI_PHI'] - [y for y in df['Tr_T_Phi']] )
    df['Tr_T_diff_eta']     = df['B_LOKI_ETA'] - [y for y in df['Tr_T_Eta'] ]
    df['Tr_T_P_proj'] = [np.vdot(x,y) for x, y  in zip(df[['B_PX', 'B_PY', 'B_PZ', 'B_PE']].values, df[['Tr_T_PX', 'Tr_T_PY', 'Tr_T_PZ', 'Tr_T_E']].values)]
    # now we go back to the b-candidate index :
    df = df.set_index([df.index,'B_len'])
    # we would like to choose the first 40 track with the highest pT (and remove the other entries per B-candidate index).
    # here is how it is done:
    # 1- simply sort the valuies 
    df = df.sort_values('Tr_T_AALLSAMEBPV',ascending=False).sort_index(level=0)
    # 2- use groupby and head combination to chop any other track which is not in the set of "first 40 highest pT"
    df = df.groupby(level=0).head(40)
    # pickle everything : 
    df.to_pickle('/eos/lhcb/user/b/bkhanji/FT/MC/Bd2JpsiKst_' + suffix + '_fltrd_'+str(i)+'.pkl')
    i = i+1
    #print(df)                                                                                                                                                
    # we are done !
    del df
    

    
    
