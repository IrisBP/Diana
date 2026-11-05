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
import roifile as roi 
import glob

######################################## functions ######################################################################

def segment_movie(img, model, T):
    '''
    use the Cellpose model to segment the BF channel and the hungarian algorithm to match the cell IDs to the previous frame 
    #output a .tif file containing all the tracked cell masks 
    '''
    cell_masks=[]
    for t in tqdm.tqdm(range(T), desc='Segmentation', leave=True):  
        BF=img[t][0]
        mask, flows, styles = model.eval([BF])
        mask=mask[0].astype('uint16')
        cell_masks.append(mask)
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
                  
def pipeline(path,path_img, path_roi,  c1, c2, gpu_status):
    '''
    Use Cellpose to segment the cells. For the cells that have been tracked by Diana (ROI files), extrack single cell information 
    Output:
    - cell masks, nuclear masks 
    - .csv file with cell information 
    '''

    model = models.CellposeModel(gpu=gpu_status)

    if 'automated_analysis' not in os.listdir(path):
        os.mkdir(path+'automated_analysis')

    list_POS=np.unique([file.split('vp=')[1].split(' type')[0] for file in os.listdir(path_roi)])
    print(len(list_POS), 'positions to analyze')


    for file in os.listdir(path_img):
        if 'MCR' in file:
            
            name=file.split('.')[0]
            print(name)
            POS=name.split('Pos')[1]
            if POS in list_POS:
                img=tif.imread(path_img+file,is_mmstack=False, is_ome=False)
                T,C,X,Y=img.shape
                print('File opened')

                if name not in os.listdir(path+'automated_analysis'):
                    os.mkdir(path+'automated_analysis/'+name)

                out_path=path+'automated_analysis/'+name+'/'
                
                if name+'_masks.tif' not in os.listdir(out_path):
                    cell_masks=segment_movie(img, model, T)
                    tif.imwrite(f'{out_path}{name}_masks.tif'.format(), np.array(cell_masks))

                print('Segmentation OK')

                if f'{name}_info.csv'.format() not in os.listdir(out_path):
                    tracked_masks=tif.imread(f'{out_path}{name}_masks.tif'.format().format())
                    dic_info={'ImgID':[], 'Frame':[], 'CellID':[], 'MaskID':[], 'Annotation':[],'Centroid_x':[], 'Centroid_y':[],'CellArea':[], 'NucleusArea':[], 'Channel_'+str(c2+1):[], 'Channel_'+str(c1+1):[]}
                    listROI_file=glob.glob(path_roi+f'cell=* vp={POS} type=nuclear.zip'.format())
                    for file in tqdm.tqdm(listROI_file, desc='ROI annotation', leave=True):
                        cellID=file.split('cell=')[1].split(' vp')[0]
                        all_nuclear_mask=[]
                        R=roi.roiread(file)
                        for r in R:
                            x=round((r.right+r.left)/2)
                            y=round((r.top+r.bottom)/2)
                            f=r.position
                            annotation=r.name
                            frame=tracked_masks[f]
                            maskID=frame[y,x]
                            if maskID!=0:
                                m=frame.copy()
                                m[m != maskID]=0
                                m[m == maskID]=1
                                ret2, nucleus_mask= cv2.threshold(m*img[f][c2],0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                                k = np.ones((3, 3), np.uint8)
                                nucleus_mask = cv2.morphologyEx(nucleus_mask, cv2.MORPH_CLOSE, k)
                                nucleus_mask = cv2.erode(nucleus_mask, k)
                                GFP=get_avg_signal(img[f], m, c2)
                                RFP=get_avg_signal(img[f], m, c1)
                                dic_info['ImgID'].append(name)
                                dic_info['Frame'].append(f)
                                dic_info['CellID'].append(cellID)
                                dic_info['MaskID'].append(maskID)
                                dic_info['Annotation'].append(annotation)
                                dic_info['CellArea'].append(m.sum())
                                dic_info['Centroid_x'].append(x)
                                dic_info['Centroid_y'].append(y)
                                dic_info['NucleusArea'].append(nucleus_mask.sum())
                                dic_info['Channel_'+str(c2+1)].append(GFP)
                                dic_info['Channel_'+str(c1+1)].append(RFP)
                            else:
                                nucleus_mask=np.zeros(frame.shape)
                            all_nuclear_mask.append(nucleus_mask)
                        tif.imwrite(f'{out_path}{name}_Nucleusmask_cell{cellID}.tif'.format(), np.array(all_nuclear_mask))
                    pd.DataFrame(dic_info).to_csv(f'{out_path}{name}_info.csv'.format(), index=False)
               
    return       
      


######################################## Scripts ######################################################################


# python MAIN.py /Volumes/biol_bc_barrallab_1/Alex/DRebholz/Experiments/Microscopy/DR_MCR1/ /Volumes/biol_bc_barrallab_1/Alex/DRebholz/Experiments/Microscopy/DR_MCR1/DR_MCR1_2/ /Volumes/biol_bc_barrallab_1/Alex/DRebholz/Experiments/Microscopy/DR_MCR1/DR_MCR1_cells/ 1 2


if __name__ == '__main__':
    path=sys.argv[1]
    path_img=sys.argv[2]
    path_roi=sys.argv[3]
    c1=int(sys.argv[4])
    c2=int(sys.argv[5])
    gpu_status=True
    pipeline(path,path_img, path_roi,  c1, c2, gpu_status)

