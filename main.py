import os
from capture_face import *
from recognize_face import *
from train_model import *

while True:

    print("For loading image in dataset, press '1' ")
    print("For recognizing face, press '2' ")
    print("To exit, press any other key\n")

    button = input("Enter: ")

    if(button=='1'):
        print("You pressed '1'")
        capture_face()
        train_model()

    elif(button=='2'):
        print("You pressed '2'")
        recognize_face()

    else:
        print("Exiting ...")
    break