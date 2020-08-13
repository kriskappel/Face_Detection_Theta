import cv2
import scipy.io as sio
import datetime as dt
import random
import numpy as np
import os
import math

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac


def face_detect(image_rgb, image_gray):
    facedetector=cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
    face=facedetector.detectMultiScale(image_gray,1.34 ,3)
    print('number of faces:')
    print(len(face))
    for x,y,z,h in face:
        cv2.rectangle(image_rgb,(x,y),(x+z,y+h),(0,0,225),3)
        #print x
        faceimg = 0

        r = max(z, h) / 2
        centerx = x + z / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = image_rgb[ny:ny+nr, nx:nx+nr]

        blob = cv2.dnn.blobFromImage(faceimg, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))


        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        #print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        label = "{},{}".format(gender, age)
        cv2.putText(image_rgb, label, (x,y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return image_rgb

def print_images(image1, image2):

    concat_images =  np.hstack((image1, image2))

    cv2.imshow("window",concat_images)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 

#SETUP
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)

mat = sio.loadmat('wiki/wiki.mat')
#END_SETUP

while(True):
    rand_img= random.randint(0,62327)
    path_to_img= str(mat['wiki'][0][0]['full_path'][0][rand_img])
    path_to_img = 'wiki/' + path_to_img[3:-2]
    print path_to_img
    print mat['wiki'][0][0]['second_face_score'][0][rand_img]
    print math.isnan(mat['wiki'][0][0]['second_face_score'][0][rand_img])

    #print mat['wiki'][0][0]['face_score'][0][rand_img]
    while(mat['wiki'][0][0]['face_score'][0][rand_img] <= 0 and math.isnan(mat['wiki'][0][0]['second_face_score'][0][rand_img])):
        print mat['wiki'][0][0]['face_score'][0][rand_img] 
        rand_img= random.randint(0,62328)
        path_to_img= str(mat['wiki'][0][0]['full_path'][0][rand_img])
        path_to_img = 'wiki/' + path_to_img[3:-2]
        print path_to_img


    my=cv2.imread(path_to_img)
    gt =cv2.imread(path_to_img)
    my2=cv2.imread(path_to_img, 0)

    gender_gt = ''
    if (mat['wiki'][0][0]['gender'][0][rand_img]) == 0:
        gender_gt= 'Female'
    elif (mat['wiki'][0][0]['gender'][0][rand_img]) == 1:
        gender_gt= 'Male'
    else:
        gender_gt= "unknown"

    date = matlab2datetime(mat['wiki'][0][0]['dob'][0][rand_img])
    age_gt = str(int(mat['wiki'][0][0]['photo_taken'][0][rand_img]) - int(date.year))
    #itemimg = random.randint(0,555)

    fls= mat['wiki'][0][0]['face_location'][0][rand_img]
    cv2.rectangle(gt,(int(fls[0][0]),int(fls[0][1])),(int(fls[0][2]),int(fls[0][3])),(255,0,0),3)


    gt_string = gender_gt + "," + age_gt 
    print gt_string

    cv2.putText(gt, gt_string, (int(fls[0][0]),int(fls[0][3])+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    my= face_detect(my, my2)
    print_images(my, gt)

