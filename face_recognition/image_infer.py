import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

if __name__ == '__main__':
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_3d_68', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = ins_get_image('t1')
    faces = app.get(img)
    tim = img.copy()
    color = (200, 160, 75)
    for face in faces:
        print(dir(face))
        print("\n")
        lmk = face.recognition
        # lmk = np.round(lmk).astype(int)
        # for i in range(lmk.shape[0]):
        #     p = tuple(lmk[i])
        #     cv2.circle(tim, p, 1, color, 1, cv2.LINE_AA)
    cv2.imwrite('./test_out.jpg', tim)

