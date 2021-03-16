from utils import config, loaddb, train, utils
import numpy as np

while True:
    print("\n************ MAIN MENU ************"
          "\n* Select mode:                    *"
          "\n*    1. Preprocess                *"
          "\n*    2. Train                     *"
          "\n*    3. Predict - not implemented *"
          "\n*    Press ENTER to exit.         *"
          "\n***********************************")
    mode = input('Option: ')

    #PREPROCESS mode
    if(mode == '1'):
        print("\n************ PREPROCESS MENU ************"
              "\n* Select database to preprocess:        *"
              "\n*    1. Columbia  - not implemented     *"
              "\n*    2. MPII                            *"
              "\n*    3. To be defined - not implemented *"
              "\n*    Hit ENTER to return to MAIN MENU   *"
              "\n*****************************************")
        preprocess_db = input('Option: ')
        
        if(preprocess_db == '1'):
            print("Preprocess Columbia\nNot implemented. Returning to MAIN MENU.")
        elif(preprocess_db == '2'):
            print("Preprocessing MPII...")
            img, gzs = utils.get_data(config.path_data)
            np.savez(config.dbpath_mpii, image=img, gaze=gzs)
            print("Done!")
        elif(preprocess_db == '3'):
            print("Preprocess TBD\nNot implemented. Returning to MAIN MENU.")
        else:
            exit
            
    #TRAIN mode
    elif(mode == '2'):
        print("\n*************** TRAIN MENU **************"
              "\n* Select database to train on:          *"
              "\n*    1. Columbia  - not implemented     *"
              "\n*    2. MPII                            *"
              "\n*    3. To be defined - not implemented *"
              "\n*    Hit ENTER to return to MAIN MENU   *"
              "\n*****************************************")
        train_db = input('Option: ')
        
        if(train_db == '1'):
            print("\nTrain on Columbia\nNot implemented. Returning to MAIN MENU.")
        elif(train_db == '2'):
            print("\nTrain on MPII")
            train.train_mpii()
        elif(train_db == '3'):
            print("\nTrain on TBD\nNot implemented. Returning to MAIN MENU.")
        else:
            exit
            
    #PREDICT mode        
    elif(mode == '3'):
        print("\n************** PREDICT MENU *************"
              "\n* Select database to predict on:        *"
              "\n*    1. Columbia  - not implemented     *"
              "\n*    2. MPII - not implemented          *"
              "\n*    3. To be defined - not implemented *"
              "\n*    Hit ENTER to return to MAIN MENU   *"
              "\n*****************************************")
        predict_db = input('Option: ')
        
        #In predict mode a new menu in which you will need to select the predictor needs to be added
        if(predict_db == '1'):
            print("Predict on Columbia\nNot implemented. Returning to MAIN MENU.")
        elif(predict_db == '2'):
            print("Predict on MPII - not implemented. Returning to MAIN MENU.")
        elif(predict_db == '3'):
            print("Predict on TBD\nNot implemented. Returning to MAIN MENU.")
        else:
            exit
    else:
        print("Exiting...")
        break