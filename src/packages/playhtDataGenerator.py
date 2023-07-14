import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython.display import Audio
import os
import time
import json
import requests

base_path = "/home/ubuntu/"


class PlayHTVoiceClone:
    # refresh token -- this is on GitHub
    def __init__(
        self, credentials_json="/home/ubuntu/configs/playht_api_credentials.json"
    ) -> None:
        with open(filepath, "r") as f:
            api_credentials = json.load(f)

        self.authorization = api_credentials["Authorization"]
        self.user_id = api_credentials["X-User-ID"]
        self.base_url = "https://play.ht/api/v1/"

        # hold urls here for now
        self.convert_url = self.base_url + "convert"
        self.download_url = self.base_url + "articleStatus"

        self.headers = api_credentials
        # temporarily set content type to json
        self.headers["Content-Type"] = "application/json"

    def select_cloned_voice(self):
        self.cloned_voices_url = self.base_url + "getClonedVoices"

        self.cloned_voice_resp = requests.get(
            self.cloned_voices_url, headers=self.headers
        )

        self.cloned_voice_id = self.cloned_voice_resp.json()["clonedVoices"][0]["id"]
        self.cloned_voice_name = self.cloned_voice_resp.json()["clonedVoices"][0][
            "name"
        ]
        print("Cloned voice name: {}".format(self.cloned_voice_name))

    def run_tts(self, text):
        tid = self._start_conversion(text)
        print("_start_conversion completed!! tid: {}".format(tid))

        audio_url = self._poll_status(tid)

        print(audio_url)

        # self._download_audio(audio_url)

    def _start_conversion(self, text):
        payload = {"voice": self.cloned_voice_id}
        payload["content"] = [text]

        convert_payload = json.dumps(payload)

        converted_voice_resp = requests.post(
            self.convert_url, headers=self.headers, data=convert_payload
        )

        return converted_voice_resp.json()["transcriptionId"]

    def _poll_status(self, tid):
        url = self.download_url + f"?transcriptionId={tid}"

        delay = 5

        print("Polling status loop started")

        while True:
            # get response
            download_resp = requests.get(url, headers=self.headers)
            # check if transcription is complete
            msg = download_resp.json().get("message")
            print(f"Messsage: {msg}")

            if msg == "Transcription completed":
                audio_url = download_resp.json().get("audioUrl")
                return audio_url
                break

            # if not, wait and try again
            print("wait and try again")
            time.sleep(delay)
