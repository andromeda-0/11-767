from capture_image import *

if __name__ == '__main__':
    for time in range(1000, 48000, 1000):
        exposure_time = (time, time + 2000)
        cam = Cam(1024, 1024, 30, exposure_time=(-1, -1))
        img = cam.capture_image()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('image' + str(exposure_time) + '.png', img)
