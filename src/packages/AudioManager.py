from pydub import AudioSegment
import os
from packages.LibrosaManager import LibrosaManager
import soundfile as sf
import librosa
import numpy as np
import random
import shutil

#utilities for converting audio files to appropriate sample rates
#and for performing adversarial laundering
class AudioManager:
    def __init__(self) -> None:
        pass

    def convertAudioDirectory(
        self,
        audio_dir: str,
        input_format: str,
        output_format: str = ".wav",
        output_dir: str = None,
        delete_original: bool = False,
        bitrate: str = None,
        codec: str = None,
    ):
        for file in os.listdir(audio_dir):
            if input_format in file:
                self.convertAudioFileTypes(
                    os.path.join(audio_dir, file),
                    output_format=output_format,
                    delete_original=delete_original,
                    output_dir=output_dir,
                    bitrate=bitrate,
                    codec=codec,
                )

    def convertAudioFileTypes(
        self,
        audio_path: str,
        output_format: str = ".wav",
        delete_original: bool = False,
        output_dir: str = None,
        output_file_name: str = None,
        bitrate: str = None,
        codec: str = None,
    ):
        assert output_format in [
            ".wav",
            ".mp4",
        ], f"{output_format} is an invalid output format. Please enter types: (.wav, .mp4)." 
        try:
            import_audio = AudioSegment.from_file(audio_path)

            if isinstance(output_file_name, type(None)):
                output_file_name = os.path.basename(audio_path)
            output_file_name = output_file_name.replace(
                os.path.splitext(output_file_name)[1], output_format
            )

            if not output_dir:
                output_dir = os.path.dirname(audio_path)

            import_audio.export(
                os.path.join(output_dir, output_file_name),
                format=output_format.replace(".", ""),
                codec=codec,
                bitrate=bitrate,
            )

            if delete_original:
                os.remove(audio_path)

        except Exception as e:
            print(f"Failed to Convert Audio File: {audio_path}")
            print("Error: ", e)

    #resampling
    def resampleAudioDirectory(
        self,
        input_directory: str,
        output_directory: str,
        target_sample_rate: int,
        replace_existing: bool = False,
    ):
        for file in os.listdir(input_directory):
            if os.path.splitext(file)[1] not in [".wav", ".mp4", ".WAV"]:
                continue

            if not replace_existing:
                if os.path.isfile(os.path.join(output_directory, file)):
                    continue

            try:
                librosa_manager = LibrosaManager(os.path.join(input_directory, file))
                resampled_audio = librosa_manager.resample(
                    target_sample_rate
                )  ## SB_Comment - see librosa manager re: resampling
                sf.write(
                    os.path.join(output_directory, file),
                    resampled_audio,
                    target_sample_rate,
                    subtype="PCM_24",
                )
            except Exception as e:
                print(f"Failed to Resample: {file}")
                print(f"Error Msg: {e}")
                print()
     
    #function for adding noise to audio
    def addNoiseWithSnr(self, audio_path: str, snr_range: list = [10, 80]):
        audio, sr = librosa.load(
            audio_path
        )

        audio_power = np.mean(audio**2)

        noise_snr = random.randint(snr_range[0], snr_range[1])
        noise_power = audio_power / (10 ** (noise_snr / 10))
        noise = np.random.normal(scale=np.sqrt(noise_power) * 100, size=len(audio))

        noisy_audio = audio + noise

        return noisy_audio, noise_snr, sr

    #adversarial laundering
    def launderAudioDirectory(
        self,
        input_dir: str,
        output_dir: str,
        noise_type: str = "random_gaussian",
        replace_existing: bool = False,
        transcode_prob=0.5,
        noise_prob=0.5,
    ):
        full_launder_details = []

        # Loop through files for laundering them
        for file in os.listdir(input_dir):

            file_launder_details = [os.path.join(input_dir, file), 0, None, 0, None]

            try:
                #get flags for laundering
                is_transcode = np.random.rand() <= transcode_prob
                is_noise = np.random.rand() <= noise_prob

                bitrate_options = ["64k", "127k", "196k"]

                #transcoding
                if is_transcode:
                    bitrate = random.choice(bitrate_options)

                    file_launder_details[1] = 1
                    file_launder_details[2] = bitrate

                    self.convertAudioFileTypes(
                        os.path.join(input_dir, file),
                        output_dir=output_dir,
                        output_format=".mp4",
                        delete_original=False,
                        bitrate=bitrate,
                        codec="aac",
                    )

                    self.convertAudioFileTypes(
                        os.path.join(output_dir, file.replace("wav", "mp4")),
                        output_format=".wav",
                        delete_original=True,
                    )

                else:
                    # if no transcode is necessary, just move the file to the new directory
                    shutil.copy(
                        os.path.join(input_dir, file), os.path.join(output_dir, file)
                    )          

                #adding noise
                if is_noise:
                    noisy_audio, noise_snr, sr = self.addNoiseWithSnr(
                        os.path.join(output_dir, file)
                    )

                    file_launder_details[3] = 1
                    file_launder_details[4] = noise_snr

                    sf.write(
                        os.path.join(output_dir, file), noisy_audio, sr
                    ) 

                full_launder_details.append(file_launder_details)

            except Exception as e:
                print(f"Failed to add noise: {file}")
                print(f"Error Msg: {e}")
                print()

        return full_launder_details
