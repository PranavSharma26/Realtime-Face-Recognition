import os
import sys
from capture_face import *
from recognize_face import *
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
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