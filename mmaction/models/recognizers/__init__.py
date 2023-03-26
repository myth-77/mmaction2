# Copyright (c) OpenMMLab. All rights reserved.
from .audio_recognizer import AudioRecognizer
from .base import BaseRecognizer
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D
from .recognizer3d_kd import Recognizer3Dkd

__all__ = ['BaseRecognizer', 'Recognizer2D', 'Recognizer3D', 'AudioRecognizer', 'Recognizer3Dkd']
