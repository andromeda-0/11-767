import argparse
import os
import re
import sys
from threading import Thread

import numpy as np
import pvrhino
import soundfile
from pvrecorder import PvRecorder
from caller_classification import *


def suppress_std(func):
    """
    Suppress warnings.
    """

    def wrapper(*args, **kwargs):
        stderr_tmp = sys.stderr
        stdout_tmp = sys.stdout
        null = open(os.devnull, 'w')
        sys.stderr = null
        sys.stdout = null
        try:
            result = func(*args, **kwargs)
            sys.stderr = stderr_tmp
            sys.stdout = stdout_tmp
            return result
        except:
            sys.stderr = stderr_tmp
            sys.stdout = stdout_tmp
            raise

    return wrapper


def mask_detection_handler():
    print("calling mask detection handler", file=sys.stderr)
    classifier_instance()


def greeting_handler():
    print("calling greeting handler")
    classifier_instance()


class RhinoDemo(Thread):
    """
    Microphone Demo for Rhino Speech-to-Intent engine. It creates an input audio stream from a microphone, monitors
    it, and extracts the intent from the speech command. It optionally saves the recorded audio into a file for further
    debugging.
    """

    def __init__(self, library_path, model_path, context_path, audio_device_index=None,
                 output_path=None):
        """
        Constructor.

        :param library_path: Absolute path to Rhino's dynamic library.
        :param model_path: Absolute path to file containing model parameters.
        :param context_path: Absolute path to file containing context model (file with `.rhn` extension). A context
        represents the set of expressions (spoken commands), intents, and intent arguments (slots) within a domain of
        interest.
        :param audio_device_index: Optional argument. If provided, audio is recorded from this input device. Otherwise,
        the default audio input device is used.
        :param output_path: If provided recorded audio will be stored in this location at the end of the run.
        """

        super(RhinoDemo, self).__init__()

        self._library_path = library_path
        self._model_path = model_path
        self._context_path = context_path

        self._audio_device_index = audio_device_index

        self._output_path = output_path
        if self._output_path is not None:
            self._recorded_frames = list()

    def run(self):
        """
         Creates an input audio stream, instantiates an instance of Rhino object, and infers the intent from spoken
         commands.
         """

        rhino = None
        recorder = None

        try:
            rhino = pvrhino.create(
                    library_path=self._library_path,
                    model_path=self._model_path,
                    context_path=self._context_path)

            recorder = PvRecorder(device_index=self._audio_device_index,
                                  frame_length=rhino.frame_length)
            recorder.start()

            print(f"Using device: {recorder.selected_device}")

            print(rhino.context_info)
            print()

            while True:
                pcm = recorder.read()

                if self._output_path is not None:
                    self._recorded_frames.append(pcm)

                is_finalized = rhino.process(pcm)
                if is_finalized:
                    inference = rhino.get_inference()
                    if inference.is_understood:
                        if inference.intent == 'maskDetection':
                            mask_detection_handler()
                        elif inference.intent == 'greeting':
                            greeting_handler()
                    else:
                        print("Didn't understand the command.\n", file=sys.stderr)

        except KeyboardInterrupt:
            print('Stopping ...')

        finally:
            if recorder is not None:
                recorder.delete()

            if rhino is not None:
                rhino.delete()

            if self._output_path is not None and len(self._recorded_frames) > 0:
                recorded_audio = np.concatenate(self._recorded_frames, axis=0).astype(np.int16)
                soundfile.write(
                        os.path.expanduser(self._output_path),
                        recorded_audio,
                        samplerate=rhino.sample_rate,
                        subtype='PCM_16')

    @classmethod
    def show_audio_devices(cls):
        devices = PvRecorder.get_audio_devices()

        for i in range(len(devices)):
            print(f'index: {i}, device name: {devices[i]}')


@suppress_std
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--context_path', help="Absolute path to context file.",
                        default='checkpoints/old_checkpoint/maskDetection_en_jetson_2021-12-18-utc_v1_6_0.rhn')

    parser.add_argument('--library_path', help="Absolute path to dynamic library.",
                        default=pvrhino.LIBRARY_PATH)

    parser.add_argument(
            '--model_path',
            help="Absolute path to the file containing model parameters.",
            default=pvrhino.MODEL_PATH)

    parser.add_argument('--audio_device_index', help='Index of input audio device.', type=int,
                        default=0)

    parser.add_argument('--output_path', help='Absolute path to recorded audio for debugging.',
                        default=None)

    parser.add_argument('--show_audio_devices', action='store_true')

    args = parser.parse_args()

    if args.show_audio_devices:
        RhinoDemo.show_audio_devices()
    else:
        if not args.context_path:
            raise ValueError('Missing path to context file')

        RhinoDemo(
                library_path=args.library_path,
                model_path=args.model_path,
                context_path=args.context_path,
                audio_device_index=args.audio_device_index,
                output_path=args.output_path).run()


#
# class Filter(object):
#     def __init__(self, stream, re_pattern):
#         self.stream = stream
#         self.pattern = re.compile(re_pattern) if isinstance(re_pattern, str) else re_pattern
#         self.triggered = False
#
#     def __getattr__(self, attr_name):
#         return getattr(self.stream, attr_name)
#
#     def write(self, data):
#         if data == '\n' and self.triggered:
#             self.triggered = False
#         else:
#             if self.pattern.search(data) is None:
#                 self.stream.write(data)
#                 self.stream.flush()
#             else:
#                 # caught bad pattern
#                 self.triggered = True
#
#     def flush(self):
#         self.stream.flush()
#

if __name__ == '__main__':
    # show audio options:
    # /usr/bin/python3 voice2intent.py --show_audio_device
    # use "index: 0, device name: PCM2902 Audio Codec Analog Mono" for our project
    # run detection on jetson
    # /usr/bin/python3 voice2intent.py  --context_path checkpoints/maskDetection_en_jetson_2021-12-18-utc_v1_6_0.rhn --audio_device_index 0
    # --output_path rec.wav
    # stdout_tmp = sys.stdout
    # stderr_tmp = sys.stderr
    # sys.stdout = Filter(sys.stdout, r'Overflow')
    # sys.stderr = Filter(sys.stderr, r'Overflow')
    classifier_instance = Classify()
    main()
    # sys.stdout = stdout_tmp
    # sys.stderr = stderr_tmp