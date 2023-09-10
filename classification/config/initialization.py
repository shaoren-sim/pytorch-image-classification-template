from classification.config.defaults import _C
from yacs.config import CfgNode


def get_default_cfg() -> CfgNode:
    """Helper function to get the default configuration file defined in classification/config/defaults."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def validate_cfg(cfg: CfgNode):
    """Validation function to ensure that specific keys defined in the CfgNode are valid.

    YACS may already provide some type safety, but specific assertions can be made to ensure that all parameters fall within expectations.
    """
    # EMA tracking cannot be active if we are not doing EMA.
    if cfg.EVALUATION.TRACK_EMA:
        assert (
            cfg.BAG_OF_TRICKS.EMA.DO_EMA
        ), "BAG_OF_TRICKS.EMA.DO_EMA must be active in order to track EMA during evaluation via EVALUATION.TRACK_EMA"

    # If the EMA model is not being selected, then there must be a dev set in order to track it.
    if cfg.EVALUATION.SELECT_EMA_MODEL:
        assert (
            cfg.EVALUATION.TRACK_EMA
        ), "EMA evaluation must be done in order to perform model selection via EMA model."
