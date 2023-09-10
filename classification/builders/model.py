import timm
from torch import nn


def build_model(
    model_name: str,
    use_pretrained: bool,
    num_classes: int,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """_summary_

    Args:
        model_name (str): Model identifier string. To use timm models, prepend "timm_" to the model name timm.list_models(). Can be extended in builders.model.
        use_pretrained (bool): Whether to use pretrained weights.
        num_classes (int): Number of image classes. Likely obtained via the Dataset's .class_to_idx property.
        dropout_rate (float, optional): Classifier dropout rate to use. Defaults to 0.0.
        drop_path_rate (float, optional): Stochastic depth path drop rate. Defaults to 0.0.

    Returns:
        nn.Module: The initialized model.
    """
    if model_name[:5] == "timm_":
        model_identifier = model_name[5:]
        print(f"Initializing timm model - {model_identifier}.")
        return timm.create_model(
            model_name=model_identifier,
            pretrained=use_pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
        )
    else:
        raise ValueError(f"Invalid model name {model_name}")
