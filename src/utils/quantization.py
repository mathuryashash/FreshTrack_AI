import torch
import torch.nn as nn
from pathlib import Path


def quantize_model(model, checkpoint_path, save_path, method="dynamic"):
    """Quantize a PyTorch model for faster inference.

    Args:
        model: The PyTorch model to quantize
        checkpoint_path: Path to the trained checkpoint
        save_path: Path to save the quantized model
        method: "dynamic" (faster inference) or "static" (requires calibration)

    Returns:
        Quantized model
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    if method == "dynamic":
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
    else:
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        quantized_model = model

    torch.save(quantized_model.state_dict(), save_path)
    print(f"Quantized model saved to {save_path}")

    return quantized_model


def get_model_size_mb(model_path):
    """Get model size in MB."""
    size_bytes = Path(model_path).stat().st_size
    return size_bytes / (1024 * 1024)


def export_to_torchscript(model, sample_input, save_path):
    """Export model to TorchScript for production deployment.

    Args:
        model: The PyTorch model to export
        sample_input: Sample input tensor for tracing
        save_path: Path to save the TorchScript model
    """
    model.eval()
    traced_model = torch.jit.trace(model, sample_input)
    traced_model.save(save_path)
    print(f"TorchScript model saved to {save_path}")
    return traced_model


def apply_torch_compile(model):
    """Apply torch.compile for PyTorch 2.0+ optimization.

    Args:
        model: The PyTorch model to compile

    Returns:
        Compiled model
    """
    if hasattr(torch, "compile"):
        return torch.compile(model, mode="reduce-overhead", backend="inductor")
    else:
        print("torch.compile not available. Using fallback.")
        return model
