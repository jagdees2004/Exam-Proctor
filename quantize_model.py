import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def quantize_mobilefacenet():
    model_path = "models/mobilefacenet.onnx"
    output_path = "models/mobilefacenet_int8.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Quantizing {model_path}...")
    
    # Use dynamic quantization for simplicity and size reduction
    # This is effective for models like MobileFaceNet when running on CPU
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8
    )
    
    initial_size = os.path.getsize(model_path) / (1024 * 1024)
    final_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Quantization complete!")
    print(f"Original size: {initial_size:.2f} MB")
    print(f"Quantized size: {final_size:.2f} MB")
    print(f"Size reduction: {((initial_size - final_size) / initial_size) * 100:.1f}%")

if __name__ == "__main__":
    quantize_mobilefacenet()
