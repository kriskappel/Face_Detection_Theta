import cv2
import scipy.io as sio
import datetime as dt
import random
import os
from matplotlib import pyplot as plt

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

# def get_img():
#     folder = random.randint(0,99)

#     if folder < 10:
#         folder = '0' + str(folder) + '/'
#     else:
#         folder = str(folder) + "/"

#     itemimg = random.choice(os.listdir("wiki/" + folder ))

#     path_img = "wiki/" + folder + itemimg

#     return path_img


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

# path_to_img = get_img()
# while(mat['wiki'][0][0]['face_score'][0][0]<=0):
#     path_to_img = get_img()

#date = matlab2datetime(mat['wiki'][0][0]['dob'][0][0])
rand_img= random.randint(0,62328)
path_to_img= str(mat['wiki'][0][0]['full_path'][0][rand_img])
path_to_img = 'wiki/' + path_to_img[3:-2]
print path_to_img


my=cv2.imread("gtb.jpeg")

my2=cv2.imread("gtb.jpeg", 0)

# my = cv2.resize(my, (640, 480))
# my2 = cv2.resize(my2, (640, 480))
#cv2.imshow("example",my2)
facedetector=cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
#facedetector=cv2.CascadeClassifier("data/haarcascades_cuda/haarcascade_frontalface_alt_tree.xml")
#facedetector=cv2.CascadeClassifier("data/vec_files/trainingfaces_24_24.vec")
#facedetector=cv2.CascadeClassifier("data/lbpcascades/lbpcascade_frontalface_improved.xml")

if (mat['wiki'][0][0]['gender'][0][rand_img]) == 0:
    print 'female'
elif (mat['wiki'][0][0]['gender'][0][rand_img]) == 1:
    print 'male'
else:
    print "unknown"

date = matlab2datetime(mat['wiki'][0][0]['dob'][0][rand_img])
print int(mat['wiki'][0][0]['photo_taken'][0][0]) - int(date.year)
#itemimg = random.randint(0,555)




face=facedetector.detectMultiScale(my2,1.3 ,3)
print('number of faces:')
print(len(face))
for x,y,z,h in face:
    cv2.rectangle(my,(x,y),(x+z,y+h),(0,0,225),3)
    #print x
    faceimg = 0

    r = max(z, h) / 2
    centerx = x + z / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)

    faceimg = my[ny:ny+nr, nx:nx+nr]

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
    cv2.putText(my, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

#cv2.imshow("window",my)



# #cv2.waitKey()
#crop_img = my[x:x+z, y:y+h]
#cv2.imshow("cropped", crop_img)

# print(my.shape[1])


# padding = 20

# bbox=[]
# bbox.append([x, y, x+z, y+h])

# face = my[max(0,bbox[1]-padding):min(bbox[3]+padding,my.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, my.shape[1]-1)]

# blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)


# faceimg = 0

# for (x, y, z, h) in face:
#     r = max(z, h) / 2
#     centerx = x + z / 2
#     centery = y + h / 2
#     nx = int(centerx - r)
#     ny = int(centery - r)
#     nr = int(r * 2)

#     faceimg = my[ny:ny+nr, nx:nx+nr]
#     #cv2.imshow("crop", faceimg)
#     #cv2.waitKey(0)



# blob = cv2.dnn.blobFromImage(faceimg, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
# genderNet.setInput(blob)
# genderPreds = genderNet.forward()
# gender = genderList[genderPreds[0].argmax()]
# print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
# label = "{}".format(gender)
# cv2.putText(my, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

# cv2.imshow("facedetective",my)

# cv2.waitKey(0)

#cv2.destroyAllWindows()



# import cv2
# my=cv2.imread(r"example1.jpg")
# my2=cv2.imread(r"example1.jpg",0)
# facedetector=cv2.CascadeClassifier()
# face=facedetector.detectMultiScale(my2,1.1,5)
# print('number of faces:')
# print(len(face))
# for x,y,z,h in face:
#     cv2.rectangle(my,(x,y),(x+z,y+h),(0,0,225),3)
# cv2.imshow("facedetective",my)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 

plt.imshow(my)
plt.show()