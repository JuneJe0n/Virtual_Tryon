import ctypes
import os
import platform
from dataclasses import dataclass
from enum import IntEnum

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from numpy.typing import NDArray


def get_resource_path(package, resource):
    """Get the path to a resource file within the package."""
    module_dir = os.path.dirname(__file__)
    return os.path.join(module_dir, resource)


def load_library():
    """
    Loads the lviton library based on the platform.
    """
    system = platform.system()
    lib_ext = {"Linux": ".so", "Darwin": ".dylib", "Windows": ".dll"}[system]
    lib_name = f"liblviton{lib_ext}"

    # Get the directory containing this file
    module_dir = os.path.dirname(__file__)

    # Try locations in order of likelihood
    possible_paths = [
        # Editable install - library in zig-out/lib/
        os.path.join(module_dir, "..", "zig-out", "lib", lib_name),
        # Installed package - library in parent directory
        os.path.join(module_dir, "..", lib_name),
        # Installed package - library in same directory
        os.path.join(module_dir, lib_name),
    ]

    for lib_path in possible_paths:
        if os.path.exists(lib_path):
            return ctypes.CDLL(lib_path)

    raise ImportError(f"LViton library not found. Searched: {possible_paths}")


class MakeupShape(IntEnum):
    FACE_BASIC = 0

    EYEBROW_BASIC = 1

    EYESHADOW_OVEREYE_FULL_BASIC = 2
    EYESHADOW_OVEREYE_CENTER_BASIC = 3
    EYESHADOW_OVEREYE_OUTER_BASIC = 4
    EYESHADOW_INNEREYE_BASIC = 5
    EYESHADOW_LOWEREYE_BASIC = 6
    EYESHADOW_LOWEREYE_TRI_BASIC = 7

    EYESHADOW_OVEREYE_SP = 8
    EYESHADOW_OVEREYE_CENTER_SP = 9
    EYESHADOW_OVEREYE_OUTER_SP = 10
    EYESHADOW_INNEREYE_SP = 11
    EYESHADOW_LOWEREYE_SP = 12

    EYESHADOW_OVEREYE_GL = 13
    EYESHADOW_OVEREYE_CENTER_GL = 14
    EYESHADOW_OVEREYE_OUTER_GL = 15
    EYESHADOW_LOWEREYE_GL = 16

    EYELINER_FILL_BASIC = 17
    EYELINER_TAIL_DOWN_SHORT_BASIC = 18

    NOSE_SHADING_FULL_BASIC = 19
    NOSE_SHADING_LONG_BASIC = 20
    NOSE_SHADING_SHORT_BASIC = 21

    BLUSHER_SIDE_WIDE_BASIC = 22
    BLUSHER_CENTER_WIDE_BASIC = 23
    BLUSHER_TOP_SLIM_BASIC = 24
    BLUSHER_GEN_Z_SIDE_BASIC = 25
    BLUSHER_GEN_Z_CENTER_BASIC = 26

    HIGHLIGHTER_EYES_BASIC = 27
    HIGHLIGHTER_CHEEKBONE_BASIC = 28
    HIGHLIGHTER_NOSE_BRIDGE_BASIC = 29
    HIGHLIGHTER_NOSETIP_BASIC = 30
    HIGHLIGHTER_FOREHEAD_BASIC = 31
    HIGHLIGHTER_EYELID_BASIC = 32
    HIGHLIGHTER_INNEREYE_BASIC = 33
    HIGHLIGHTER_CHINTIP_BASIC = 34

    LIP_FULL_BASIC = 35
    LIP_THIN_BASIC = 36

    FACEMESH_TESSELATION = 37


@dataclass(init=True, repr=True)
class MakeupOptions:
    shape: MakeupShape
    color: tuple[int, int, int]
    alpha: int
    sigma: int
    gamma: int = 0


class LViton:
    def __init__(
        self,
        lib_path: str = "",
        face_landmarker_path: str = "",
    ):
        self.image: NDArray[np.uint8] | None = None
        self.landmarks: NDArray[np.float32] | None = None
        self.detected: bool = False
        if face_landmarker_path != "":
            self._init_landmarks(face_landmarker_path)
        else:
            self._init_landmarks(get_resource_path("lviton", "face_landmarker.task"))
        if lib_path == "":
            self._lib = load_library()
        else:
            self._lib = ctypes.CDLL(lib_path)

        # Set up JXL functions if available
        try:
            self._lib.load_jxl_image.restype = ctypes.POINTER(ctypes.c_uint8)
        except AttributeError:
            # JXL support not available in this build
            pass

        # Set up PNG functions
        try:
            self._lib.load_png_image.restype = ctypes.POINTER(ctypes.c_uint8)
        except AttributeError:
            # PNG support not available in this build
            pass

    def print_version(self):
        self._lib.print_version()

    def set_image(
        self, image: NDArray[np.uint8], landmarks: NDArray[np.float32] = None
    ) -> bool:
        """
        Computes the landmarks for the given image.
        The return value indicates whether a face was found in the image.
        """
        assert len(image.shape) >= 2 and len(image.shape) <= 3
        # make sure we are working with RGBA images
        if len(image.shape) == 2 or image.shape[2] == 1:
            self.image = np.dstack(
                (image, image, image, 255 * np.ones(image.shape[:2]))
            ).astype(np.uint8)
        elif image.shape[2] == 2:
            self.image = np.dstack(
                (image, image, 255 * np.ones(image.shape[:2]))
            ).astype(np.uint8)
        elif image.shape[2] == 3:
            self.image = np.dstack((image, 255 * np.ones(image.shape[:2]))).astype(
                np.uint8
            )
        elif image.shape[2] == 4:
            self.image = image.astype(np.uint8)
        else:
            raise ValueError("Wrong image shape: " + str(image.shape))

        if landmarks is None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=self.image)
            face_landmarks = self.face_landmarker.detect(mp_image).face_landmarks
            self.detected = len(face_landmarks) != 0
            if not self.detected:
                self.image = None
                return False

            # allocate the landmarks: MediaPipe defines 478 landmarks
            assert len(face_landmarks[0]) == 478
            self.num_landmarks = len(face_landmarks[0])
            landmarks = [0] * 2 * self.num_landmarks

            # fill the landmarks manually
            for i, p in enumerate(face_landmarks[0]):
                landmarks[i * 2 + 0] = p.x
                landmarks[i * 2 + 1] = p.y
            self.landmarks = np.array(landmarks, dtype=np.float32)
        else:
            assert len(landmarks) == 478
            self.landmarks = np.array(landmarks, dtype=np.float32)
            self.detected = True
        return self.detected

    def _init_landmarks(self, face_landmarker_path: str):
        base_options = python.BaseOptions(model_asset_path=face_landmarker_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def extract_aligned_face(self, size: int = 256, padding: float = 0.25):
        """
        Return the aligned/cropped RGBA face as a numpy array of (size, size, 4)
        """
        rows, cols = self.image.shape[:2]
        out = np.zeros((size, size, 4)).astype(np.uint8)
        self._lib.extract_aligned_face(
            ctypes.c_void_p(self.image.ctypes.data),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(cols),
            ctypes.c_void_p(out.ctypes.data),
            ctypes.c_size_t(size),
            ctypes.c_size_t(size),
            ctypes.c_float(padding),
            ctypes.c_void_p(self.landmarks.ctypes.data),
            ctypes.c_size_t(478),
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
        )
        return out

    def apply_makeup(
        self,
        makeup_options: list[MakeupOptions],
        mirror: bool = False,
        split: float = 0,
    ):
        assert self.image is not None and self.landmarks is not None
        split = max(0, min(1, split))
        rows, cols = self.image.shape[:2]
        num_options = 7
        options_size = num_options * len(makeup_options)
        options = np.zeros(options_size, dtype=np.uint8)

        for i, option in enumerate(makeup_options):
            assert len(option.color) == 3
            options[i * num_options + 0] = option.shape
            options[i * num_options + 1] = max(0, min(255, option.color[0]))
            options[i * num_options + 2] = max(0, min(255, option.color[1]))
            options[i * num_options + 3] = max(0, min(255, option.color[2]))
            options[i * num_options + 4] = max(0, min(255, option.alpha))
            options[i * num_options + 5] = max(0, min(255, option.sigma))
            options[i * num_options + 6] = max(0, min(255, option.gamma))

        makeup = self.image.copy()
        self._lib.apply_makeup(
            ctypes.c_void_p(makeup.ctypes.data),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(cols),
            ctypes.c_void_p(self.landmarks.ctypes.data),
            ctypes.c_bool(mirror),
            ctypes.c_float(split),
            ctypes.c_void_p(options.ctypes.data),
            ctypes.c_size_t(len(makeup_options)),
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
        )
        return np.array(makeup[:, :, :3], dtype=np.uint8)

    def load_jxl(self, filename: str) -> NDArray[np.uint8]:
        if not hasattr(self._lib, "load_jxl_image"):
            raise RuntimeError(
                "JXL support not available in this build. Use a regular image format instead."
            )

        rows = ctypes.c_size_t()
        cols = ctypes.c_size_t()
        channels = ctypes.c_size_t()
        data_ptr = self._lib.load_jxl_image(
            ctypes.c_char_p(filename.encode("utf-8")),
            ctypes.byref(rows),
            ctypes.byref(cols),
            ctypes.byref(channels),
        )
        if not data_ptr:
            raise RuntimeError("Failed to load JXL image")
        image_data = np.ctypeslib.as_array(
            data_ptr, shape=(rows.value, cols.value, channels.value)
        )
        return image_data

    def save_jxl(self, image: NDArray[np.uint8], filename: str, quality: float = 90):
        if not hasattr(self._lib, "save_jxl_image"):
            raise RuntimeError(
                "JXL support not available in this build. Use a regular image format instead."
            )

        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        rows, cols, channels = image.shape
        self._lib.save_jxl_image(
            ctypes.c_void_p(image.ctypes.data),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(cols),
            ctypes.c_size_t(channels),
            ctypes.c_char_p(filename.encode("utf-8")),
            ctypes.c_float(quality),
        )

    def load_png(self, filename: str) -> NDArray[np.uint8]:
        rows = ctypes.c_size_t()
        cols = ctypes.c_size_t()
        channels = ctypes.c_size_t()
        data_ptr = self._lib.load_png_image(
            ctypes.c_char_p(filename.encode("utf-8")),
            ctypes.byref(rows),
            ctypes.byref(cols),
            ctypes.byref(channels),
        )
        if not data_ptr:
            raise RuntimeError("Failed to load PNG image")
        image_data = np.ctypeslib.as_array(
            data_ptr, shape=(rows.value, cols.value, channels.value)
        )
        return image_data

    def save_png(self, image: NDArray[np.uint8], filename: str):
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        rows, cols, channels = image.shape
        self._lib.save_png_image(
            ctypes.c_void_p(image.ctypes.data),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(cols),
            ctypes.c_size_t(channels),
            ctypes.c_char_p(filename.encode("utf-8")),
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from os import makedirs
    from os.path import basename, exists, join, splitext

    parser = ArgumentParser("LViton")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="output directory"
    )
    parser.add_argument(
        "-s", "--settings", metavar="u8", nargs=5, type=int, help="r g b a s"
    )
    parser.add_argument("-n", "--name", type=str, help="file name", default=None)
    parser.add_argument(
        "input", metavar="INPUT", nargs="+", type=str, help="path to image"
    )
    args = parser.parse_args()

    lviton = LViton(lib_path="zig-out/lib/liblviton.so")
    lviton.print_version()
    for input in args.input:
        if not exists(args.output):
            makedirs(args.output, mode=0o755)

        r, g, b, a, s = args.settings
        options = [
            MakeupOptions(MakeupShape.LIP_FULL_BASIC, (r, g, b), a, s, 5),
        ]
        image = lviton.load_jxl(input)

        if args.name is None:
            filename = splitext(basename(input))[0]
        else:
            filename = args.name

        if not lviton.set_image(image):
            print("no faces detected")
            exit(1)
        result = lviton.apply_makeup(options)
        lviton.save_jxl(result, join(args.output, filename) + ".jxl")
