from pathlib import Path

#For AutoKeras
database_path = '' #used as a global variable database selector
output_path = '' #used as a global variable output directory selector
#--------------------------------------------------------------------#

#Datapath for Normalized MPII
path_data = './databases/mpii/Data/Normalized'

#Databases path
dbpath_cave_ak = Path('./databases/columbia')
dbpath_mpii_ak = Path('./databases/mpii/Data/Original')

dbpath_mpii = './databases/mpii/out.npz'
dbpath_tbd = './databases/tbd'

#Output path
outpath_cave = Path('./output/cave')
outpath_mpii = Path('./output/mpii')

#Training config
epochs=10
batch_size=64
max_trials=2 #for AutoKeras
split=5000 #for AutoKeras

#Img settings
img_height = 135
img_width = 300
depth = 1

#For AllClassic
myModelType = 'model.classic'
myModelName = 'MobileNetV2'

img_height = 36
img_width = 60
weights={'imagenet', None}
weights=None
trainable = {True, False}
trainable = True
AUG = True
PREPROC = True