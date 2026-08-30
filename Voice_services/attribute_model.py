import time

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


MODEL_NAME = (
    "audeering/wav2vec2-large-robust-6-ft-age-gender"
)

class RegressionHead(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.dropout = nn.Dropout(
            config.final_dropout
        )

        self.out_proj = nn.Linear(
            config.hidden_size,
            1
        )

    def forward(self, x):

        x = self.dropout(x)

        x = self.dense(x)

        x = torch.tanh(x)

        x = self.dropout(x)

        x = self.out_proj(x)

        return x

class ClassificationHead(nn.Module):

    def __init__(
        self,
        config,
        num_labels=3
    ):

        super().__init__()

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size
        )

        self.dropout = nn.Dropout(
            config.final_dropout
        )

        self.out_proj = nn.Linear(
            config.hidden_size,
            num_labels
        )

    def forward(self, x):

        x = self.dropout(x)

        x = self.dense(x)

        x = torch.tanh(x)

        x = self.dropout(x)

        x = self.out_proj(x)

        return x

class AgeGenderModel(
    Wav2Vec2PreTrainedModel
):

    def __init__(self, config):

        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(
            config
        )

        self.age = RegressionHead(
            config
        )

        self.gender = ClassificationHead(
            config,
            num_labels=3
        )

        self.post_init()

    def forward(
        self,
        input_values,
        attention_mask=None
    ):

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        pooled = torch.mean(
            hidden_states,
            dim=1
        )

        age = self.age(
            pooled
        )

        gender = self.gender(
            pooled
        )

        return age, gender

class AttributeInference:

    def __init__(self):

        print(
            "Loading age/gender model..."
        )


        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        self.processor = (
            Wav2Vec2Processor.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            AgeGenderModel.from_pretrained(
                MODEL_NAME
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Age/gender model loaded."
        )

    def predict(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000
    ):

        start_time = time.perf_counter()

        if audio is None:

            raise ValueError(
                "Audio is None"
            )

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        audio = np.squeeze(
            audio
        )

        if audio.ndim != 1:

            raise ValueError(
                f"Expected 1D audio, "
                f"got {audio.shape}"
            )

        if audio.size == 0:

            raise ValueError(
                "Audio is empty"
            )

        print(
            "Inference audio shape:",
            audio.shape
        )

        print(
            "Inference audio duration:",
            round(
                audio.size / sampling_rate,
                2
            ),
            "seconds"
        )

        max_value = np.max(
            np.abs(audio)
        )

        if max_value > 0:

            audio = (
                audio / max_value
            )
        audio_quality = (
            self.get_audio_quality(
                audio
            )
        )


        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        input_values = (
            inputs.input_values
            .to(self.device)
        )

        attention_mask = (
            inputs.get(
                "attention_mask"
            )
        )

        if attention_mask is not None:

            attention_mask = (
                attention_mask.to(
                    self.device
                )
            )

        with torch.no_grad():

            age_output, gender_output = (
                self.model(
                    input_values,
                    attention_mask=attention_mask
                )
            )
        age_value = (
            age_output
            .squeeze()
            .item()
        )

        estimated_age = (
            age_value * 100
        )


        estimated_age = max(
            0,
            min(
                estimated_age,
                100
            )
        )

        gender_probabilities = (
            torch.softmax(
                gender_output,
                dim=-1
            )
        )

        gender_probabilities = (
            gender_probabilities
            .squeeze()
            .cpu()
            .numpy()
        )

        labels = [
            "unknown",
            "female",
            "male"
        ]

        gender_index = int(
            np.argmax(
                gender_probabilities
            )
        )

        gender = labels[
            gender_index
        ]

        gender_confidence = float(
            gender_probabilities[
                gender_index
            ]
        )

        age_bracket = (
            self.get_age_bracket(
                estimated_age
            )
        )

        age_confidence = (
            self.calculate_age_confidence(
                estimated_age
            )
        )

        processing_ms = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        return {

            "gender": gender,

            "gender_confidence": round(
                gender_confidence,
                3
            ),

            "age_bracket": age_bracket,

            "age_confidence": round(
                age_confidence,
                3
            ),

            "processing_ms": processing_ms,

            "audio_quality": audio_quality
        }

    @staticmethod
    def get_age_bracket(
        age: float
    ):

        if age < 18:

            return "unknown"

        if age <= 30:

            return "18-30"

        if age <= 45:

            return "31-45"

        if age <= 60:

            return "46-60"

        return "60+"

    @staticmethod
    def calculate_age_confidence(
        age: float
    ):
        """
        Simple heuristic confidence.

        This is NOT calibrated model
        probability. For production,
        calibrate this using validation data.
        """

        if age < 18:

            return 0.40

        if age <= 30:

            return 0.63

        if age <= 45:

            return 0.63

        if age <= 60:

            return 0.63

        return 0.60

    @staticmethod
    def get_audio_quality(
        audio: np.ndarray
    ):

        if (
            audio is None
            or audio.size == 0
        ):

            return "insufficient"

        rms = np.sqrt(
            np.mean(
                np.square(audio)
            )
        )


        if rms < 0.005:

            return "insufficient"


        if rms < 0.02:

            return "degraded"


        clipping_ratio = np.mean(
            np.abs(audio) >= 0.99
        )

        if clipping_ratio > 0.05:

            return "degraded"

        return "good"


attribute_inference = AttributeInference()