import numpy as np
import cv2
import datetime
from skimage import feature 
from matplotlib import pyplot as plt


def face_detect(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    frontal_cascade=cv2.CascadeClassifier('haar_features/haarcascade_frontalface_default.xml')
    profile_cascade=cv2.CascadeClassifier('haar_features/haarcascade_profileface.xml')
    #glasses_cascade=cv2.CascadeClassifier('haar_features/haarcascade_eye_tree_eyeglasses.xml')
    
    frontal_face=frontal_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=3)
    profile_face=profile_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=3)
    reverse_profile_face=profile_cascade.detectMultiScale(cv2.flip(img,1),scaleFactor=1.5,minNeighbors=3)

    #glasses_face=glasses_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    frontal_img=save_face(img,frontal_face)
    profile_img=save_face(img,profile_face)
    reverse_profile_img=save_face(img,reverse_profile_face,reverse=False)
    #reverse_profile_img=0
    #save_face(img,glasses_face)
    return frontal_face,profile_face,reverse_profile_face,frontal_img,profile_img,reverse_profile_img


def save_face(frame,faces,reverse=False):
    imgs=[]
    added_pixels=0
    for i in faces:
        if reverse:
            interest_region=frame[i[1]-added_pixels:i[1]+i[3]+added_pixels,-i[0]+i[2]-added_pixels:-i[0]+2*i[2]+added_pixels]
        else:
            interest_region=frame[i[1]-int(added_pixels/2):i[1]+i[3]+int(added_pixels/2),i[0]-int(added_pixels/2):i[0]+i[2]+int(added_pixels/2)]
        img_path="data/"+str(datetime.datetime.now())+".jpg"
        #cv2.imwrite(img_path,interest_region)
        imgs.append(interest_region)
    return imgs


def draw_bbox(frame,faces,text=False,reverse=False):
    color=(0,0,255)
    stroke=2
    added_pixels=100

    for i in faces:
       
        # if reverse:
        #     h=i[1]+i[3]+added_pixels
        #     w=i[0]+i[2]+added_pixels
        #     cv2.rectangle(frame,(i[0],i[1]),(w,h),color,stroke)
        #     if text:
        #         cv2.putText(frame, text, (-i[0]+i[2], i[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)


        # else:
        h=i[1]+i[3]+int(added_pixels/2)
        w=i[0]+i[2]+int(added_pixels/2)
        cv2.rectangle(frame,(i[0]-int(added_pixels/2),i[1]-int(added_pixels/2)),(w,h),color,stroke)

        if text:
            cv2.putText(frame, text, (i[0], i[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        


def video_cap():
    # 0 is for cam1, 1 for 2 and so on
    cap = cv2.VideoCapture(0)

    # run main loop
    # if we show one image after antoher, it becomes video
    while True:
        frame =cv2.imread("real_me2.png")         #cap.read()          # read from camera
        frontal_face,profile_face,reverse_profile_face,_,_,_=face_detect(frame)
        draw_bbox(frame,frontal_face,text="frontal")
        draw_bbox(frame,profile_face,text="side")
        draw_bbox(frame,reverse_profile_face,reverse=True,text="reverse")
        

        cv2.imshow("press Q to exit",frame)         # show image
        if cv2.waitKey(10) == ord('q'):  # wait a bit, and see keyboard press
            break                        # if q pressed, quit

    # release things before quiting
    cap.release()
    cv2.destroyAllWindows()









def main():
    video_cap()

if __name__ == '__main__':
    main()
