# Yeast analysis pipeline for Diana


## Prerequisite: 
- python 
- miniconda 

## To install the pipeline: 
- Download the pipeline: git clone https://github.com/IrisBP/Diana.git
- navigate to folder: cd Diana 
- Create conda environment: conda env create -f env.yml

## To run the pipeline: 
- Activate the conda environment: conda activate Diana
- python MAIN.py path_to_img_folder 1 2 

the 2 numbers indicate the order of the fluorescent channels: 
- 1 2 = red channel then green channel 
- 2 1 = green channel then red channel 

>>> This pipeline currently designed to run using a GPU 
>>> to run on CPU, change GPU_status to False in Main.py (line 144)
