import dlib
import face_detect as fd
import cv2
import numpy as np
from scipy.spatial import distance as dist





def video_cap():
    # 0 is for cam1, 1 for 2 and so on
    cap = cv2.VideoCapture(0)

    frontal_img=0
    profile_img=0
    reverse_profile_img=0 
    frontal_text=0 
    profile_text=0
    reverse_profile_text=0
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    blinks=0
    while True:
        ret, frame = cap.read()          # read from camera
        if not ret:
            continue
        frontal_face,profile_face,reverse_profile_face,frontal_img,profile_img,reverse_profile_img=fd.face_detect(frame)
        
        for j in frontal_face:

            d=dlib.rectangle(int(j[0]),int(j[1]),int(j[0])+int(j[2]),int(j[1])+int(j[3]))
            landmarks=predictor(frame,d)
            for k in range(36, 48):
                x=landmarks.part(k).x
                y=landmarks.part(k).y
                cv2.circle(frame,(x,y),3,(255,0,0),-1)
                cv2.putText(frame, str(k), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36,255,12), 2)


        left=np.divide(dist.euclidean((landmarks.part(41).x,landmarks.part(41).y),(landmarks.part(37).x,landmarks.part(37).y)) + dist.euclidean((landmarks.part(40).x,landmarks.part(40).y),(landmarks.part(38).x,landmarks.part(38).y)),2*dist.euclidean((landmarks.part(39).x,landmarks.part(39).y),(landmarks.part(36).x,landmarks.part(36).y)))
        right=np.divide(dist.euclidean((landmarks.part(47).x,landmarks.part(47).y),(landmarks.part(43).x,landmarks.part(43).y)) + dist.euclidean((landmarks.part(46).x,landmarks.part(46).y),(landmarks.part(44).x,landmarks.part(44).y)),2*dist.euclidean((landmarks.part(45).x,landmarks.part(45).y),(landmarks.part(42).x,landmarks.part(42).y)))
        print(left)
        print(right)
        if right  < 0.3 or left < 0.3:
            blinks="yes"
        else:
            blinks="no"
        text="Blinks: "+str(blinks)
        fd.draw_bbox(frame,frontal_face,text=text)


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
