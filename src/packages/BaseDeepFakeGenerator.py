from lib2to3.pgen2.tokenize import tokenize
import os
from secrets import token_urlsafe
import pandas as pd

#base class used by other generators to load text from a dataframe or directory 
#and process transcripts
class BaseDeepFakeGenerator:
    def __init__(self, tokenize_type: str = None):
        if not isinstance(tokenize_type, type(None)):
            assert tokenize_type.lower() in [
                "word",
                "sentence",
            ], "If you provide a tokenize type, it must be sentence or word"
        self.tokenize_type = tokenize_type

    def loadTextFromDataFrame(
        self,
        dataframe_path: str,
        source_col: str,
        transcript_col: str,
        punc_to_remove: list = None,
    ):
        metadata = pd.read_csv(dataframe_path)
        source_paths = list(metadata[source_col])
        file_names = [os.path.basename(source_path) for source_path in source_paths]
        transcripts = list(metadata[transcript_col])

        if punc_to_remove:
            transcripts = self.process_transcripts(transcripts, punc_to_remove)

        return file_names, transcripts

    def loadTextFromDirectory(self, dir_name: str):
        for file_name in os.listdir(dir_name):
            if ".txt" in file_name:
                pass

    def _readTextFile(self, text_path: str):
        with open(text_path) as f:
            lines = f.readlines()

        f.close()

    def process_transcripts(self, transcripts: list, punc_to_remove: list):
        processed_transcripts = []
        for idx, transcript in enumerate(transcripts):
            for punc in punc_to_remove:
                print(idx)
                transcript = transcript.replace(punc, "")
            processed_transcripts.append(transcript)
        return processed_transcripts
