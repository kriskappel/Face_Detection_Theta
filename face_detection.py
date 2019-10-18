import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']


genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

facedetector=cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")


def img_to_cv(img):
    bridge=CvBridge()

	cv_image = bridge.imgmsg_to_cv2(img, "bgr8")

	detect_face(cv_image)


def detect_face(frame):

my=cv2.imread(r"example3.jpg")
my2=cv2.imread(r"example3.jpg",0)
#cv2.imshow("example",my2)
#facedetector=cv2.CascadeClassifier("data/haarcascades_cuda/haarcascade_frontalface_alt_tree.xml")
#facedetector=cv2.CascadeClassifier("data/vec_files/trainingfaces_24_24.vec")
#facedetector=cv2.CascadeClassifier("data/lbpcascades/lbpcascade_frontalface_improved.xml")


face=facedetector.detectMultiScale(my2,1.05 ,3)
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
    label = "{}".format(gender)
    cv2.putText(my, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)




cv2.imshow("facedetective",my)



cv2.waitKey(0)
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

cv2.destroyallWindows()



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
# cv2.waitKey(0)
# cv2.destroyallWindows() 