import config
from utils import loaddb, train, utils
import numpy as np

def main_menu():
    print("\n************ MAIN MENU ************"
          "\n* Select mode:                    *"
          "\n*    1. Preprocess                *"
          "\n*    2. Train                     *"
          "\n*    3. Predict - not implemented *"
          "\n*    Press ENTER to exit.         *"
          "\n***********************************")
          
def process_menu():
    print("\n************ PREPROCESS MENU ************"
          "\n* Select database to preprocess:        *"
          "\n*    1. Columbia  - not implemented     *"
          "\n*    2. MPII                            *"
          "\n*    3. To be defined - not implemented *"
          "\n*    Hit ENTER to return to MAIN MENU   *"
          "\n*****************************************")
          
def train_menu():
    print("\n*************** TRAIN MENU **************"
          "\n* Select database to train on:          *"
          "\n*    1. Columbia                        *"
          "\n*    2. MPII                            *"
          "\n*    3. To be defined - not implemented *"
          "\n*    Hit ENTER to return to MAIN MENU   *"
          "\n*****************************************")
          
def train_method_menu():
    print("\n************** METHOD MENU *************"
          "\n* Select method to train with:         *"
          "\n*    1. Manual                         *"
          "\n*    2. AllClassic                     *"
          "\n*    3. AutoKeras                      *"
          "\n*    4. AutoKeras Regression           *"
          "\n*    Hit ENTER to return to MAIN MENU  *"
          "\n****************************************")
          
def predict_menu():
    print("\n************** PREDICT MENU *************"
          "\n* Select database to predict on:        *"
          "\n*    1. Columbia  - not implemented     *"
          "\n*    2. MPII - not implemented          *"
          "\n*    3. To be defined - not implemented *"
          "\n*    Hit ENTER to return to MAIN MENU   *"
          "\n*****************************************")

while True:
    #MAIN MENU mode
    main_menu()
    mode = input('Option: ')

    #PREPROCESS mode
    if(mode == '1'):
        process_menu()
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
        train_menu()
        train_db = input('Option: ')
        
        if(train_db == '1'):
            print("\nTrain on Columbia")
            config.database_path = config.dbpath_cave_ak
            config.output_path = config.outpath_cave
            train.train_ak()
        elif(train_db == '2'):
            print("\nTrain on MPII")
            config.database_path = config.dbpath_mpii_ak
            config.output_path = config.outpath_mpii
            
            #MPII train method
            train_method_menu()
            train_method = input('Option: ')
            
            if(train_method == '1'):
                train.train_mpii_manual()
            elif(train_method == '2'):
                train.train_mpii_classic() 
            elif(train_method == '3'):
                train.train_ak()
            elif(train_method == '4'):
                train.train_ak_regression()
            else:
                exit
                
        elif(train_db == '3'):
            print("\nTrain on TBD\nNot implemented. Returning to MAIN MENU.")
        else:
            exit

    #PREDICT mode        
    elif(mode == '3'):
        predict_menu()
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
