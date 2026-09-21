"""Speech workload contract; real execution is planned."""

from workloads.deferred import DeferredWorkloadEngine
from utils.constants import WORKLOAD_SPEECH


class SpeechWorkload(DeferredWorkloadEngine):
    workload_name = WORKLOAD_SPEECH
    model_family = "Whisper"
    input_type = "audio"
    task_types = ("automatic_speech_recognition",)
