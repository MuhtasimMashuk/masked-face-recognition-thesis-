
import face_recognition
import cv2
import numpy as np
import os
import glob

# Get a reference to webcam #0 (the default one)
webcam_video_stream = cv2.VideoCapture(0)

#make array of sample pictures with encodings
known_face_encodings = []
known_face_names = []
#dirname = os.path.dirname(__file__)  #disable kore dekhci
#path = os.path.join(dirname, 'known_people/')#same

#make an array of all the saved jpg files' paths
list_of_files = [f for f in glob.glob('known_people/'+'*.jpg')]
#find number of known faces
number_files = len(list_of_files)

names = list_of_files.copy()

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    known_face_encodings.append(globals()['image_encoding_{}'.format(i)])

    # Create array of known names
    names[i] = names[i].replace("known_people/", "")  
    known_face_names.append(names[i])

all_face_locations = []
all_face_encodings = []
all_face_names = []


while True:
   
    ret,current_frame = webcam_video_stream.read()
    
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
 
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='cnn')
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations,num_jitters=5)


    
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
      
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
     
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
       
   
        name_of_person = 'Unknown face'
        
      
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (0,255,0),1)
    
    
    cv2.imshow("Webcam Video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()        