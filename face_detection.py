import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from std_msgs.msg import Int16

flag_detect = False


def img_to_cv(img):
    global flag_detect
    if flag_detect :   
        bridge=CvBridge()

    	cv_image = bridge.imgmsg_to_cv2(img, "bgr8")

    	detect_face(cv_image)

        flag_detect = False
    else:
        pass

def detect_face(frame):

    #========default variables=======
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    genderList = ['Male', 'Female']

    genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)

    facedetector=cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
    #========

    faces_pub = rospy.Publisher('face_detection/number', Int16, queue_size=10)
    male_pub = rospy.Publisher('face_detection/male', Int16, queue_size=10)
    female_pub = rospy.Publisher('face_detection/female', Int16, queue_size=10)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #my=cv2.imread(r"example3.jpg")
    #my2=cv2.imread(r"example3.jpg",0)
#cv2.imshow("example",my2)
#facedetector=cv2.CascadeClassifier("data/haarcascades_cuda/haarcascade_frontalface_alt_tree.xml")
#facedetector=cv2.CascadeClassifier("data/vec_files/trainingfaces_24_24.vec")
#facedetector=cv2.CascadeClassifier("data/lbpcascades/lbpcascade_frontalface_improved.xml")

    face=facedetector.detectMultiScale(frame_gray,1.05 ,3)
    print('number of faces:')
    print(len(face))
    
    number_of_faces=len(face)
    number_of_males = 0
    number_of_females = 0

    for x,y,z,h in face:
        #cv2.rectangle(my,(x,y),(x+z,y+h),(0,0,225),3)
#print x
        faceimg = 0

        r = max(z, h) / 2
        centerx = x + z / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = frame[ny:ny+nr, nx:nx+nr] #TODO ver se é colored ou nao

        blob = cv2.dnn.blobFromImage(faceimg, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        if gender == 'male':
            number_of_males = number_of_males + 1
        else:
            number_of_females = number_of_females + 1

        #label = "{}".format(gender)
        #cv2.putText(my, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    faces_pub.publish(number_of_faces)
    male_pub.publish(number_of_males)
    female_pub.publish(number_of_females)
#cv2.imshow("facedetective",my)

#cv2.waitKey(0)

#cv2.destroyallWindows()

def analyze():
    global flag_detect
    flag_detect = True

if __name__ == "__main__":
    rospy.init_node("face_detector", anonymous = True) 

    rospy.Subscriber("camera/rgb/image_raw", Image, img_to_cv)

    rospy.Subscriber("analyze_frame", Bool, analyze)

    rospy.spin()

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