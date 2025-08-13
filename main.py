from sys import argv

from lviton import LViton, MakeupOptions, MakeupShape
from PIL import Image
import numpy as np


def main():
    lviton = LViton(
        lib_path="/home/jiyoon/LViton/lib/liblviton-x86_64-linux-3.0.3.so",
        face_landmarker_path="model/face_landmarker.task",
    )
    makeup_options = [
        MakeupOptions(
            shape=MakeupShape.LIP_THIN_BASIC,
            color=(192, 0, 0),
            alpha=128,
            sigma=32,
            gamma=1,
        ),
        MakeupOptions(
            shape=MakeupShape.EYESHADOW_OVEREYE_FULL_BASIC,
            color=(0, 128, 0),
            alpha=50,
            sigma=50,
            gamma=100,
        ),
        MakeupOptions(
            shape=MakeupShape.EYESHADOW_OVEREYE_CENTER_SP,
            color=(10,20,40),
            alpha=100,
            sigma=50,
            gamma=100,
        ),
        MakeupOptions(
            shape=MakeupShape.BLUSHER_CENTER_WIDE_BASIC,
            color=(128, 0, 64),
            alpha=50,
            sigma=224,
            gamma=100,
        ),
        MakeupOptions(
            shape=MakeupShape.FACE_BASIC,
            color=(224, 192, 224),
            alpha=72,
            sigma=224,
            gamma=10,
        ),
        MakeupOptions(
            shape=MakeupShape.NOSE_SHADING_LONG_BASIC,
            color=(128,52,60),
            alpha=80,
            sigma=50,
        ),
        MakeupOptions(
            shape=MakeupShape.EYEBROW_BASIC,
            color=(0,0,128),
            alpha=80,
            sigma=10,   
        ),
        MakeupOptions(
            shape=MakeupShape.EYELINER_TAIL_DOWN_SHORT_BASIC,
            color=(80,80,50),
            alpha=255,
            sigma=0,
            gamma=10
        )
    ]

    if len(argv) < 2:
        raise ValueError("Provide an image")

    image = np.array(Image.open(argv[1]))
    print(image.shape)
    if lviton.set_image(image):
        result = lviton.apply_makeup(makeup_options)
        lviton.save_png(result, "/home/jiyoon/LViton/test/results/2.png")


if __name__ == "__main__":
    main()
