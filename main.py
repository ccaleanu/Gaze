from utils import config, loaddb, train, utils
import numpy as np

while True:
    print("Select mode:\n1. Preprocess\n2. Train\n3. Predict - not implemented\n")
    mode = input()

    #PREPROCESS mode
    if(mode == '1'):
        print("Select database to PREPROCESS: \n"
              "0. Quit\n"
              "1. Columbia  - not implemented\n"
              "2. MPII\n"
              "3. To be defined - not implemented\n")
        preprocess_db = input()
        
        if(preprocess_db == '1'):
            print("Preprocess Columbia\nNot implemented. Quitting...")
        elif(preprocess_db == '2'):
            print("Preprocessing MPII...")
            img, gzs = utils.get_data(config.path_data)
            np.savez(config.dbpath_mpii, image=img, gaze=gzs)
            print("Done!")
        elif(preprocess_db == '3'):
            print("Preprocess TBD\nNot implemented. Quitting...")
        else:
            print("Quitting")
            exit
            
    #TRAIN mode
    elif(mode == '2'):
        print("Select database to TRAIN on: \n"
              "0. Quit\n"
              "1. Columbia  - not implemented\n"
              "2. MPII\n"
              "3. To be defined - not implemented\n")
        train_db = input()
        
        if(train_db == '1'):
            print("Train on Columbia\nNot implemented. Quitting...")
        elif(train_db == '2'):
            print("Train on MPII")
            train.train_mpii()
        elif(train_db == '3'):
            print("Train on TBD\nNot implemented. Quitting...")
        else:
            print("Quitting")
            exit
            
    #PREDICT mode        
    elif(mode == '3'):
        print("PREDICT mode is not implemented yet.")
        print("Select database to PREDICT on: \n"
              "0. Quit\n"
              "1. Columbia  - not implemented\n"
              "2. MPII\n"
              "3. To be defined - not implemented\n")
        predict_db = input()
        
        if(predict_db == '1'):
            print("Predict on Columbia\nNot implemented. Quitting...")
        elif(predict_db == '2'):
            print("Predict on MPII - not implemented")
        elif(predict_db == '3'):
            print("Predict on TBD\nNot implemented. Quitting...")
        else:
            print("Quitting")
            exit
    else:
        print("Exiting...")
        break