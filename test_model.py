import torch
import torch.nn as nn
from torchsummary import summary
from model import Model_1, Model_2, Model_4

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_architecture():
    """
    Target: Verify model architectures and parameter counts
    Result: Successfully validates models under specified parameter limits
    Analysis:
    - Confirms parameter count for each model
    - Verifies correct model inheritance
    - Tests forward pass functionality
    - Provides detailed model summaries
    - Ensures output shape matches requirements (1, 10)
    """
    
    # Test Model_1
    model1 = Model_1()
    assert isinstance(model1, nn.Module), "Model should be a PyTorch Module"
    
    # Check parameter count for Model_1
    param_count1 = count_parameters(model1)
    print(f"Model_1 Parameter Count: {param_count1}")
    assert param_count1 <= 8000, f"Model_1 has {param_count1} parameters, exceeding 8000 limit"
    
    # Test Model_2
    model2 = Model_2()
    assert isinstance(model2, nn.Module), "Model should be a PyTorch Module"
    
    # Check parameter count for Model_2
    param_count2 = count_parameters(model2)
    print(f"Model_2 Parameter Count: {param_count2}")
    assert param_count2 <= 8000, f"Model_2 has {param_count2} parameters, exceeding 8000 limit"
    
    # Test Model_4
    model4 = Model_4()
    assert isinstance(model4, nn.Module), "Model should be a PyTorch Module"
    
    # Check parameter count for Model_4
    param_count4 = count_parameters(model4)
    print(f"Model_4 Parameter Count: {param_count4}")
    assert param_count4 <= 9600, f"Model_4 has {param_count4} parameters, exceeding 9600 limit"
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST image size
    output1 = model1(dummy_input)
    output2 = model2(dummy_input)
    output4 = model4(dummy_input)
    
    assert output1.shape == (1, 10), f"Expected output shape (1, 10), got {output1.shape}"
    assert output2.shape == (1, 10), f"Expected output shape (1, 10), got {output2.shape}"
    assert output4.shape == (1, 10), f"Expected output shape (1, 10), got {output4.shape}"
    
    # Print model summaries
    print("\nModel_1 Summary:")
    summary(model1, (1, 28, 28), device="cpu")
    
    print("\nModel_2 Summary:")
    summary(model2, (1, 28, 28), device="cpu")
    
    print("\nModel_4 Summary:")
    summary(model4, (1, 28, 28), device="cpu")

if __name__ == "__main__":
    test_model_architecture()
    print("All tests passed!") 