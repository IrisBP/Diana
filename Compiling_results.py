
import numpy as np 
import pandas as pd 
import os 
import tqdm 
import sys
import glob


# python Compiling_results.py /Volumes/biol_bc_barrallab_1/Alex/DRebholz/Experiments/Microscopy/DR_MCR1/automated_analysis
if __name__ == '__main__':
    path=sys.argv[1]
    all_files=[]
    for file in glob.glob(f'{path}/*/*_info.csv'.format()):
        df=pd.read_csv(file)
        all_files.append(df)
        name=file.split('/')[-1].split('_MMS')[0]
    df=pd.concat(all_files)
    df.to_csv(path+'/'+name+'_results.csv', index=False)



