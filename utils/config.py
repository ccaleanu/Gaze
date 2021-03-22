from pathlib import Path

#Datapath for Normalized MPII
path_data = './databases/mpii/Data/Normalized'

#Databases path
dbpath_cave = Path('./databases/columbia')
dbpath_mpii = './databases/mpii/out.npz'
dbpath_tbd = './databases/tbd'

#Output path
outpath_cave = Path('./output/cave')

#Training config
epochs=10
batch_size=64
max_trials=2 #for AutoKeras

#Img size (for columbia)
img_height = 135
img_width = 300
