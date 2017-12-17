#colaboration Statement
#Rene Borr
#R00805596
#On my honor, I have not given, nor received, nor witnessed any unauthorized assistance on this work.
#I worked on this assignment alone, and refered to:
    #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    #https://github.com/kunalgupta777/OpenCV-Face-Filters/blob/master/filters.py
    #https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
    #http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    #https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    
#Source images
    #https://giphy.com/gifs/loop-galaxy-s1P4kzgXdyZK8
    #http://www.clker.com/clipart-black-glasses.html

import numpy as np
import cv2
import os

width = 1280
height = 720
"""
main

The main function does not take any arguments or return anything.
Run the main statement to see live snapchat filter.
"""
def main():
    #Load all of the images from a folder into a list 
    #(This will be used as our overlay later)
    video = load_images_from_folder("Galaxy")
    
    
    
    #Load other images into variables
    glasses = cv2.imread('glasses_outer.tif',cv2.IMREAD_UNCHANGED)
    lens = cv2.imread('glasses_inner.tif',cv2.IMREAD_UNCHANGED)    
    lens = make_lens_transparent(lens)
    
    #Load haarcascade for feature recognition
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    #Create a video capture variable
    cap = cv2.VideoCapture(0)

    #Create counter variable
    #(this will be used to tell what overlay image from video we should use)
    count = 0
    
    #Create infanite loop
    while(True):
        
        #Read current frame from webcam
        ret, frame = cap.read()
        
        #print instructions for user
        cv2.putText(frame,"Hold any key to play. Press 'q' to quit", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        
        #create a grey version of the frame
        #use it to find the location of the faces in the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        #Set variable overlay to the corect image in video
        overlay = video[count]
        
        #loop through all the faces in the frame
        for (x,y,w,h) in faces: 
            #Place the galsses on their faces
            put_glasses(glasses,lens,overlay,frame,x,y,w,h)
        
        #show the frame
        
        cv2.imshow('filter',frame)
        
        #if the user is pressing the "q" key, stop video feed
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        #incrament count
        #if count has reached its limit, reset count
        count +=1
        if(count>23):
            count = 0
            
    #Destroy all windows and shut off camera    
    cap.release()
    cv2.destroyWindow('filter')
    
    
    
"""
make_transparent

this function acts as a green screen, and sets the alpha value to zero 
anywhere the photo is completly green

Args:
        image (numpy.ndarray): A image represented in a numpy array.
        
Returns:
        image (numpy.ndarray): A image represented in a numpy array, but now 
        with the approprate alpha values.
"""
        
        
def make_lens_transparent(image):
    
    rows, cols = image.shape[:2]
   
    #Loop through all pixels in the array
    # if the pixel has a high enough green value, make it transparent
    for r in range(rows):
        for c in range(cols):
            if(image[r][c][1] >= 240):
                image[r][c][3] = 0
            else:
                image[r][c][3]= 175


    return image

"""
add_alpha

this function takes a regualar image, and adds an alpha channell so it can be transparent

Args:
        image (numpy.ndarray): A image represented in a numpy array.
        
Returns:
        image (numpy.ndarray): A image represented in a numpy array, but now with an alpha channell
"""

"""
put_glasses

Adds the glasses, and its lens, to the proper location in the frame.

Args:
        glasses (numpy.ndarray): A image represented in a numpy array. used as the outer part of the glasses
        lens (numpy.ndarray):  A image represented in a numpy array. Used as the "lens" section of the glasses
        overlay (numpy.ndarray):  A image represented in a numpy array. an image representation of the colors used for the "lens" section
        fc(numpy.ndarray):  A image represented in a numpy array. The live video caputred from the camera
        
        x(int): the x location of the face in the frame
        y(int): the y location of the face in the frame
        w(int): the width of the face in the frame
        h(int): the height of the face in the frame
        
"""
def put_glasses(glasses,lens,overlay,fc,x,y,w,h):
    
    #set the height and width of the glasses according to the face dimentions
    g_width = w
    g_height = int(h/2)
    #resize the images accordingly
    glasses = cv2.resize(glasses,(g_width,g_height))
    lens = cv2.resize(lens,(g_width,g_height))
    
    #set the x, and the y offset from the top corner of the face
    x_offset=x
    y_offset=int(y + w/6)
    
    #calculate x and y values for the specific offsets
    y1, y2 = y_offset, y_offset + glasses.shape[0]
    x1, x2 = x_offset, x_offset + glasses.shape[1]
    
    #set up the lens with the correct colors
    lens = set_up_lens(lens,overlay,x_offset,y_offset)
    
    #Add the glasses, and the lens to the frame
    if(x_offset + g_width<width and y_offset+g_height<height ):
        
        alpha_s = glasses[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        for c in range(0, 3):
            fc[y1:y2, x1:x2, c] = (alpha_s * glasses[:, :, c] + alpha_l * fc[y1:y2, x1:x2, c])
            
        alpha_s = lens[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        for c in range(0, 3):
            fc[y1:y2, x1:x2, c] = (alpha_s * lens[:, :, c] + alpha_l * fc[y1:y2, x1:x2, c])
      
        
"""
set_up_lens
replaces the glasses with only specific colors needed

Args:
        lens (numpy.ndarray): A image represented in a numpy array. representative of the "lens" in the glasses
        overlay(numpy.ndarray): A image represented in a numpy array. Representative of the colors that will replace the lens
        x(int):the x location of the glasses in the frame
        y(int):the y location of the glasses in the frame

"""
def set_up_lens(lens,overlay,x,y):
    
    rows,cols = lens.shape[:2]

    lens[:,:,2] = overlay[x:x+rows,y:y+cols][:,:,2]
    lens[:,:,1] = overlay[x:x+rows,y:y+cols][:,:,1]
    lens[:,:,0] = overlay[x:x+rows,y:y+cols][:,:,0]      
    
    return lens

"""
load_images_from_folder
reads all images from a specified folder and places them in a list

Args: 
    folder(String): A string representation of the folder name

Returns:
    images(list of numpy.ndarray): a list of images from the specified folder
"""

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,filename))
        if image is not None:
            image = cv2.resize(image,(width,width))
            images.append(image)
    return images

#Run the main statement    
main()

"""
Below are other methods for placement of other types of items in the frame that I didnt end up using.

def place_static(frame,image,x,y):
    
    y1, y2 = y, y + image.shape[0]
    x1, x2 = x, x + image.shape[1]
    
    alpha_s = image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    
    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * image[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
    
    return frame

def put_ears(ears,fc,x,y,w,h):
    
    face_width = w
    face_height = h
    
    ear_width = int(face_width*.75)
    ear_height = int(face_height*.3)
    
    ears = cv2.resize(ears,(ear_width,ear_height))
    
    x_offset=x + int(face_width*.1)
    y_offset=int(y-face_height*.25)
    
    y1, y2 = y_offset, y_offset + ears.shape[0]
    x1, x2 = x_offset, x_offset + ears.shape[1]
    

    alpha_s = ears[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        fc[y1:y2, x1:x2, c] = (alpha_s * ears[:, :, c] + alpha_l * fc[y1:y2, x1:x2, c])

    return fc

def put_nose(nose,fc,x,y,w,h):
    
    face_width = w
    face_height = h
    
    nose_width = int(face_width*.4)
    nose_height = int(face_height*.25)
    
    nose = cv2.resize(nose,(nose_width,nose_height))
    
    x_offset= int(x + face_width/2 - nose_width/2)
    y_offset= int(y + face_height/2)
    
    y1, y2 = y_offset, y_offset + nose.shape[0]
    x1, x2 = x_offset, x_offset + nose.shape[1]
    

    alpha_s = nose[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        fc[y1:y2, x1:x2, c] = (alpha_s * nose[:, :, c] + alpha_l * fc[y1:y2, x1:x2, c])

    return fc

def put_hat(hat,fc,x,y,w,h):
    
    face_width = w
    face_height = h
    
    
    
    hat_width = face_width+3
    hat_height = int(0.35*face_height)+3
    
    hat = cv2.resize(hat,(hat_width,hat_height))
    
    x_offset=x
    y_offset=int(y-hat_height/2)
    
    y1, y2 = y_offset, y_offset + hat.shape[0]
    x1, x2 = x_offset, x_offset + hat.shape[1]
    

    alpha_s = hat[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    
    for c in range(0, 3):
        fc[y1:y2, x1:x2, c] = (alpha_s * hat[:, :, c] + alpha_l * fc[y1:y2, x1:x2, c])

    return fc
"""
