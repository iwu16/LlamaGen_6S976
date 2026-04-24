# watermark package — CGZ scheme for LlamaGen
from watermark.cgz_watermark import (
    VOCAB_SIZE,
    DEFAULT_CONTEXT_LEN,
    get_green_mask,
    cgz_sample_from_logits,
    compute_threshold,
    compute_detection_score,
    detect,
    cgz_generate,
)
