import torch
from pathlib import Path
from models_factory import ModifEffNetB0, ModifEffNetB3, ModifMobileNet, ResNet18Pothole, ModifResNet18, ModifConvNeXt

def get_model_shell(model_name: str):
    """
    Identifies which class to instantiate based on the filename.
    """
    if "95.44_resnet18" in model_name:
        return ResNet18Pothole()
    elif "97.14_modif_resnet18" in model_name:
        return ModifResNet18()
    elif "97.36_modif_convnext" in model_name:
        return ModifConvNeXt()
    elif "95.20_modif_effnet_b0" in model_name:
        return ModifEffNetB0()
    elif "94.00_modif_effnet_b3" in model_name:
        return ModifEffNetB3()
    elif "95.20_modif_mobilenet" in model_name:
        return ModifMobileNet()
    return None

def load_all_models(folder_path: str = "save_models", device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_map = {}
    path = Path(folder_path).resolve()
    
    if not path.exists():
        print(f"Error: Folder '{path}' not found.")
        return model_map

    # Check for both .pth and _checkpoint files
    for model_path in list(path.glob("*.pth")) + list(path.glob("*_checkpoint")):
        model_name = model_path.stem
        
        try:
            model = get_model_shell(model_name)
            
            if model is None:
                print(f"Skipping {model_name}: No matching class found in factory.")
                continue

            # 2. Load the state dictionary
            state_dict = torch.load(model_path, map_location=device)

            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # 4. Load weights into the shell
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval() 
            
            model_map[model_name] = model
            print(f"Successfully loaded weights into: {model_name}")
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            
    return model_map