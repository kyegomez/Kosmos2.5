import torch
from torch import nn
from zeta import (
    AutoregressiveWrapper,
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)

class Kosmos(nn.Module):
    def __init__(
        self,
        image_size = 256,
        patch_size = 32,
        encoder_dim = 512,
        encoder_depth = 6,
        encoder_heads=8,
        num_tokens = 108481,
        max_seq_len = 8192,
        decoder_dim = 2560,
        decoder_depth = 32,
        decoder_heads = 24,
        use_abs_pos_emb = False,
        alibi_pos_bias=True,
        alibi_num_heads=12,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
    ):
        super(Kosmos, self).__init__()

        self.encoder = ViTransformerWrapper(
            image_size = image_size,
            patch_size = patch_size,
            attn_layers = Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads
            )
        )

        self.decoder = Transformer(
            num_token=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=True,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            )
        )

        self.decoder = AutoregressiveWrapper(self.decoder)
    
    def forward(self, img, text):
        try:
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forwrd method: {error}")
            raise
        