# Gaze

Added the code without the database.  
  
Edit config.py and modify the database path.  
Run main.py and first choose to preprocess the database to be able to train afterwards. Preprocess needs to be run only once.
  
There are improvements to be made and uploaded to the actual code.  
  
   
      
Description:  
main.py: code to be run. A simple menu from which you select what you want to do. It calls the preprocess, train and predict functions.  
databases/  
   Folder that contains the databases  
   -- /mpii: contains the mpii database  
   -- /columbia: contains the columbia database  
   -- /tbd: a to be defined database  
output/  
   Folder that will contain the output training and prediction information  
utils/  
   The separate and useful code is stored here   
   -- config.py: configuration variables: training parameters, paths to databases  
   -- loaddb.py: custom functions to load the databases  
   -- models.py: custom defined models  
   -- train.py: training functions, where the models are loaded and actually trained  
   -- utils.py: useful functions: convert gaze points, custom loss functions, reading files functions  
  
   
!!! For MPII AutoKeras