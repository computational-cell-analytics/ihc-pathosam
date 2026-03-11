import argparse
import os

import timm
import tifffile
import torch
import zarr
from torchvision import transforms


def get_model_path(model_folder="/mnt/lustre-grete/usr/u12086/models/hugging-face/mahmood-lab/uni-v2"):
    filename = "pytorch_model.bin"
    model_path = os.path.join(model_folder, filename)
    if os.path.exists(model_path):
        return model_path

    from huggingface_hub import login, hf_hub_download
    login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
    os.makedirs(model_folder, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/UNI2-h", filename=filename, local_dir=model_folder, force_download=True)
    return model_path


def get_uni_model_and_transform(device):
    model_path = get_model_path()
    timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True
            }
    model = timm.create_model(pretrained=False, **timm_kwargs)
    model.to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model.eval()
    return model, transform


def apply_uni(model, transform, input_path, output_path):
    # TODO
    tif = tifffile.TiffFile(input_path)
    store = tif.aszarr()
    group = zarr.open(store, mode="r")
    data = groups[str(0)]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    # parser.add_argument("-p", "--pattern")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = get_uni_model_and_transform(device=device)

    apply_uni(model, transform, args.input_path, args.output_path)


if __name__ == "__main__":
    main()
