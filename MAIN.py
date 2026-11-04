######################################## Cell segmentation and Intensity analysis ####################################
# Made by Iris Barbier - November 2026 
# CLI: python MAIN.py path_to_file 1 2 (if c2=RFP and c3=GFP) or  python MAIN.py path_to_file 2 1 (if c2=GFP and c3=RFP)

######################################## imports ######################################################################
import tifffile as tif 
import numpy as np 
import pandas as pd 
import cv2 
from cellpose import models, io
import matplotlib.pyplot as plt 
import os 
import tqdm 
import hungarian as hu
from multiprocessing import Pool, cpu_count
import sys

######################################## functions ######################################################################

def segment_track_movie(img, model, T):
    '''
    use the Cellpose model to segment the BF channel and the hungarian algorithm to match the cell IDs to the previous frame 
    #output a .tif file containing all the tracked cell masks 
    '''
    cell_masks=[]
    for t in tqdm.tqdm(range(T), desc='Segmentation', leave=True):  
        BF=img[t][0]
        mask, flows, styles = model.eval([BF])
        mask=mask[0].astype('uint16')
        if t==0:
            cell_masks.append(mask)
        if t>0:
            prev=cell_masks[t-1]
            corrected=hu.correspondence(prev, mask)
            corrected=corrected.astype('uint16')
            cell_masks.append(corrected)
    return cell_masks

def get_avg_signal(img, mask, channel):
    '''
    Get the distribution of intensity values within a masked image and return the average value 
    '''
    applied_mask=mask*img[channel]
    dist=np.array([i for i in applied_mask.ravel() if i!=0])
    if len(dist)>0:
            return dist.mean()
    else:
        return 0

def get_cell_info(V):
    '''
    For each cell: 
    - get the nuclear mask using Otsu thresholding
    - get cell and nuclear size (pixel number)
    - get average intensity in the 2 fluorescent channels 
    '''
    img, mask,c1, c2, cell=V
    m=mask.copy()
    m[m != cell]=0
    m[m == cell]=1
    ret2, nucleus_mask= cv2.threshold(m*img[c2],0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_CLOSE, k)
    nucleus_mask = cv2.erode(nucleus_mask, k)
    GFP=get_avg_signal(img, m, c2)
    RFP=get_avg_signal(img, m, c1)
    return nucleus_mask, m.sum(), nucleus_mask.sum(), GFP, RFP 


def pipeline(path, c1, c2, gpu_status):
    '''
    Use Cellpose to segment the cells and the hungarian algorithm to track the masks. Extrack single cell information \
    Output:
    - tracked cell masks, tracked nuclear masks 
    - .csv file with cell information 
    '''

    model = models.CellposeModel(gpu=gpu_status)

    if 'automated_analysis' not in os.listdir(path):
        os.mkdir(path+'automated_analysis')

    for file in os.listdir(path):
        if 'MCR' in file:
            name=file.split('.')[0]
            print(name)
            img=tif.imread(path+file,is_mmstack=False, is_ome=False)
            T,C,X,Y=img.shape
            print('File opened')

            if name not in os.listdir(path+'automated_analysis'):
                os.mkdir(path+'automated_analysis/'+name)

            out_path=path+'automated_analysis/'+name+'/'
            
            if name+'_masks.tif' not in os.listdir(out_path):
                cell_masks=segment_track_movie(img, model, T)
                tif.imwrite(f'{out_path}{name}_masks.tif'.format(), np.array(cell_masks))
            print('Segmentation OK')
            if f'{name}_info.csv'.format() not in os.listdir(out_path):
                tracked_masks=tif.imread(f'{out_path}{name}_masks.tif'.format().format())
                thread_count=cpu_count()
                dic_info={'ImgID':[], 'Frame':[], 'Cell':[], 'CellArea':[], 'NucleusArea':[], 'Channel_'+str(c2+1):[], 'Channel_'+str(c1+1):[]}
                all_nuclear_mask=[]
                for t in tqdm.tqdm(range(T), desc='Description', leave=True):  
                    mask=tracked_masks[t]
                    frame=img[t]
                    nucleus_masks=np.zeros(mask.shape)
                    with Pool(thread_count) as p:
                        list_inputs=[(frame, mask, c1, c2 , i) for i in np.unique(mask) if i!=0]
                    
                        infos=p.map(get_cell_info, list_inputs)
                        for i in range(len(infos)):
                            nucleus_mask, Msum, Nsum, GFP, RFP =infos[i]
                            cell=list_inputs[i][4]

                            nucleus_masks+=nucleus_mask*mask
                            dic_info['ImgID'].append(name)
                            dic_info['Frame'].append(t)
                            dic_info['Cell'].append(cell)
                            dic_info['CellArea'].append(Msum)
                            dic_info['NucleusArea'].append(Nsum)
                            dic_info['Channel_'+str(c2+1)].append(GFP)
                            dic_info['Channel_'+str(c1+1)].append(RFP)
                        
                    all_nuclear_mask.append(nucleus_masks)
                    
            
                pd.DataFrame(dic_info).to_csv(f'{out_path}{name}_info.csv'.format(), index=False)
                tif.imwrite(f'{out_path}{name}_nuclear_masks.tif'.format(), np.array(all_nuclear_mask))

    return               
                        


######################################## Scripts ######################################################################


if __name__ == '__main__':
   
    path=sys.argv[1]
    c1=int(sys.argv[2])
    c2=int(sys.argv[3])
    gpu_status=True
    pipeline(path, c1, c2, gpu_status)
