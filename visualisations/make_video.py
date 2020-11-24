import cv2
import glob
import os

save_dir = os.path.join('../p1_samples/case_5')

video_name = os.path.join(save_dir, 'video.avi')

images = [img for img in sorted(os.listdir(save_dir)) if img.endswith(".png") and img.startswith("vis")]
frame = cv2.imread(os.path.join(save_dir, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(save_dir, image)))

# cv2.destroyAllWindows()
video.release()