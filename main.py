import cv2 


# define a video capture object 
vid = cv2.VideoCapture(0) 

while(True):
    ret, frame = vid.read() 

    # Display the resulting frame 
    cv2.imshow('frame', frame) 

    # press 'q' to close the video window
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 