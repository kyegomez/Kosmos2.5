import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor

logger = getLogger(__name__)

class Tokenizer:
    def __init__(
        self,
        model_path: str
    ):
        assert os.path.isfile(model_path), model_path

        self.sp_model = SentencePieceProcessor(model_path=model_path)
        logger.info(f"Reloaded SentiencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        
        self.img_token = "<img>"
        self.img_token_close = "</img>"
        self.img_token_id = self.sp_model.piece_to_id(self.img_token)
        self.img_token_close_id = self.sp_model.piece_to_id(self.img_token_close)



        logger.info(
            f"#Words: {self.n_words}, - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(
        self,
        s: str,
        bos: bool,
        eos: bool
    ) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t
    
    def decode(
        self,
        t: List[int]
    ) -> str:
        return self.sp_model.decode(t)