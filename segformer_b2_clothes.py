import os
import numpy as np
# from urllib.request import urlopen # Unused
# import torchvision.transforms as transforms # Unused
import folder_paths
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image # ImageOps, ImageFilter were unused directly
import torch.nn as nn
import torch

# --- Lazy Loading Globals ---
SEGFORMER_PROCESSOR = None
SEGFORMER_MODEL = None
SEGFORMER_MODEL_LOADED = False
MODEL_FOLDER_PATH = os.path.join(folder_paths.models_dir, "segformer_b2_clothes")

def _load_segformer_model_if_needed():
    global SEGFORMER_PROCESSOR, SEGFORMER_MODEL, SEGFORMER_MODEL_LOADED
    
    if SEGFORMER_MODEL_LOADED:
        return

    print(f"[SegformerB2Clothes] Attempting to lazy load model from: {MODEL_FOLDER_PATH}")
    
    if not os.path.isdir(MODEL_FOLDER_PATH):
        print(f"[SegformerB2Clothes] ERROR: Model folder not found at {MODEL_FOLDER_PATH}. Please ensure the model is correctly placed.")
        # SEGFORMER_MODEL_LOADED remains False, so subsequent calls will fail.
        # You might want to raise an exception here to make the failure more explicit.
        raise FileNotFoundError(f"[SegformerB2Clothes] Model folder not found: {MODEL_FOLDER_PATH}")

    try:
        SEGFORMER_PROCESSOR = SegformerImageProcessor.from_pretrained(MODEL_FOLDER_PATH)
        SEGFORMER_MODEL = AutoModelForSemanticSegmentation.from_pretrained(MODEL_FOLDER_PATH)
        SEGFORMER_MODEL_LOADED = True
        print("[SegformerB2Clothes] Model and processor loaded successfully via lazy loading.")
    except Exception as e:
        print(f"[SegformerB2Clothes] ERROR: Failed to load model/processor during lazy load: {e}")
        # Re-raise the exception so the workflow execution clearly fails if model loading fails.
        raise e

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_segmentation(tensor_image):
    _load_segformer_model_if_needed() # Ensure model is loaded

    # The check below is now more robust as _load_segformer_model_if_needed will raise an error if loading fails.
    # However, keeping a direct check can be a safeguard.
    if not SEGFORMER_MODEL_LOADED or SEGFORMER_PROCESSOR is None or SEGFORMER_MODEL is None:
        # This state should ideally not be reached if _load_segformer_model_if_needed raises on failure.
        raise RuntimeError(f"[SegformerB2Clothes] Model and/or processor not loaded despite attempt. Check logs and model path: {MODEL_FOLDER_PATH}")

    cloth = tensor2pil(tensor_image)
    inputs = SEGFORMER_PROCESSOR(images=cloth, return_tensors="pt")
    outputs = SEGFORMER_MODEL(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg,cloth

class segformer_b2_clothes:
   
    def __init__(self):
        # REMOVED: _load_segformer_model_if_needed() 
        # Model will now load only when get_segmentation is first called.
        pass 
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {     
                 "image":("IMAGE", {"default": "","multiline": False}),
                 "Face": ("BOOLEAN", {"default": True, "label_on": "enabled脸部", "label_off": "disabled"}),
                 "Hat": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Hair": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Upper_clothes": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Skirt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Pants": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Dress": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Belt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "shoe": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "leg": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "arm": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Bag": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "Scarf": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask_image",)
    FUNCTION = "sample"
    CATEGORY = "CXH"
    OUTPUT_NODE = True

    def sample(self,image,Face,Hat,Hair,Upper_clothes,Skirt,Pants,Dress,Belt,shoe,leg,arm,Bag,Scarf):
        results = []
        for item_tensor in image:
            pred_seg, cloth = get_segmentation(item_tensor) # This will trigger model load if not already loaded
            
            labels_to_keep = [0] 
            if not Hat: labels_to_keep.append(1)
            if not Hair: labels_to_keep.append(2)
            # Note: Sunglasses (label 3) are not controlled by a flag here.
            # If you want to control sunglasses, you'd need a "Sunglasses" boolean input.
            if not Upper_clothes: labels_to_keep.append(4)
            if not Skirt: labels_to_keep.append(5)
            if not Pants: labels_to_keep.append(6)
            if not Dress: labels_to_keep.append(7)
            if not Belt: labels_to_keep.append(8)
            if not shoe:
                labels_to_keep.append(9)
                labels_to_keep.append(10)
            if not Face: labels_to_keep.append(11)
            if not leg:
                labels_to_keep.append(12)
                labels_to_keep.append(13)
            if not arm:
                labels_to_keep.append(14) 
                labels_to_keep.append(15) 
            if not Bag: labels_to_keep.append(16)
            if not Scarf: labels_to_keep.append(17)
                
            mask_array = np.isin(pred_seg, labels_to_keep).astype(np.uint8) 
            output_mask_image = Image.fromarray(mask_array * 255).convert("RGB")
            results.append(pil2tensor(output_mask_image))

        return (torch.cat(results, dim=0),)

# It's good practice to ensure necessary external modules are importable at the top.
# If folder_paths is critical for MODEL_FOLDER_PATH, ensure it's robustly available.
# ComfyUI ensures folder_paths is available when it loads custom nodes.
# import os
# import numpy as np
# from urllib.request import urlopen
# import torchvision.transforms as transforms  
# import folder_paths
# from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
# from PIL import Image,ImageOps, ImageFilter
# import torch.nn as nn
# import torch

# # comfy_path = os.path.dirname(folder_paths.__file__)
# # custom_nodes_path = os.path.join(comfy_path, "custom_nodes")


# # 指定本地分割模型文件夹的路径
# model_folder_path = os.path.join(folder_paths.models_dir,"segformer_b2_clothes")

# processor = SegformerImageProcessor.from_pretrained(model_folder_path)
# model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)

# def tensor2pil(image):
#     return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# # Convert PIL to Tensor
# def pil2tensor(image):
#     return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# # 切割服装
# def get_segmentation(tensor_image):
#     cloth = tensor2pil(tensor_image)
#     # 预处理和预测
#     inputs = processor(images=cloth, return_tensors="pt")
#     outputs = model(**inputs)
#     logits = outputs.logits.cpu()
#     upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
#     pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
#     return pred_seg,cloth


# class segformer_b2_clothes:
   
#     def __init__(self):
#         pass
    
#     # Labels: 0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes", 5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf"
    
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {"required":
#                 {     
#                  "image":("IMAGE", {"default": "","multiline": False}),
#                  "Face": ("BOOLEAN", {"default": True, "label_on": "enabled脸部", "label_off": "disabled"}),
#                  "Hat": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Hair": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Upper_clothes": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Skirt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Pants": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Dress": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Belt": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "shoe": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "leg": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "arm": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Bag": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                  "Scarf": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
#                 }
#         }

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("mask_image",)
#     OUTPUT_NODE = True
#     FUNCTION = "sample"
#     CATEGORY = "CXH"

#     def sample(self,image,Face,Hat,Hair,Upper_clothes,Skirt,Pants,Dress,Belt,shoe,leg,arm,Bag,Scarf):
        
#         results = []
#         for item in image:
        
#             # seg切割结果，衣服pil
#             pred_seg,cloth = get_segmentation(item)
#             labels_to_keep = [0]
#             # if background :
#             #     labels_to_keep.append(0)
#             if not Hat:
#                 labels_to_keep.append(1)
#             if not Hair:
#                 labels_to_keep.append(2)
#             if not Upper_clothes:
#                 labels_to_keep.append(4)
#             if not Skirt:
#                 labels_to_keep.append(5)
#             if not Pants:
#                 labels_to_keep.append(6)
#             if not Dress:
#                 labels_to_keep.append(7)
#             if not Belt:
#                 labels_to_keep.append(8)
#             if not shoe:
#                 labels_to_keep.append(9)
#                 labels_to_keep.append(10)
#             if not Face:
#                 labels_to_keep.append(11)
#             if not leg:
#                 labels_to_keep.append(12)
#                 labels_to_keep.append(13)
#             if not arm:
#                 labels_to_keep.append(14) 
#                 labels_to_keep.append(15) 
#             if not Bag:
#                 labels_to_keep.append(16)
#             if not Scarf:
#                 labels_to_keep.append(17)
                
#             mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
            
#             # 创建agnostic-mask图像
#             mask_image = Image.fromarray(mask * 255)
#             mask_image = mask_image.convert("RGB")
#             mask_image = pil2tensor(mask_image)
#             results.append(mask_image)

#         return (torch.cat(results, dim=0),)
