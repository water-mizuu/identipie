# SOURCE: https://stackoverflow.com/questions/59948996/how-to-use-webcam-as-a-screen-of-pygame

import cv2
import pygame
import numpy as np
import time

pygame.init()
pygame.display.set_caption("OpenCV camera stream on Pygame")
#0 Is the built in camera
vcap = cv2.VideoCapture(0)

# get vcap property 
camera_width  = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
camera_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# These are the dimensions of the window.
window_width = int(camera_width * 1.2)
window_height = int(camera_height * 1.2)

surface = pygame.display.set_mode([window_width, window_height])

#Gets fps of your camera
fps = vcap.get(cv2.CAP_PROP_FPS)

print("fps:", fps)
#If your camera can achieve 60 fps
#Else just have this be 1-30 fps
vcap.set(cv2.CAP_PROP_FPS, 30)

stamp = time.time()
accumulative_deltas = [0.0]

while True:
    try:
        # Necessary for the OS to know it's not hanging up.
        pygame.event.pump()

        current_stamp = time.time() 
        delta = current_stamp - stamp
        accumulative_deltas = [d + delta for d in accumulative_deltas]

        surface.fill([0, 0, 0])

        success, frame = vcap.read()
        if not success:
            break

        if accumulative_deltas[0] > 100.0:
            print(accumulative_deltas, frame[0][0])
            accumulative_deltas[0] = 0.0

        #for some reasons the frames appeared inverted
        frame = np.fliplr(frame)
        frame = np.rot90(frame)

        # The video uses BGR colors and PyGame needs RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.resize(frame,
                           dsize=(window_height, window_width), 
                           interpolation=cv2.INTER_CUBIC)

        # Assuming this is an RGB matrix of [width] × [height], we can
        #   send this image to the model.

        surf = pygame.surfarray.make_surface(frame)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                vcap.release()
                pygame.quit()
                break

        # Show the PyGame surface!
        surface.blit(surf, (0,0))
        pygame.display.flip()

    except KeyboardInterrupt:
        print("Program terminated")
        break