import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    return img


class SAM:
    def __init__(self, model_type: str, model_path: str, device: str = "cuda"):
        self.__model_type = model_type
        self.__model_path = model_path
        self.__device = device

        sam = sam_model_registry[self.__model_type](checkpoint=self.__model_path)
        sam.to(device=self.__device)

        self.__mask_generator = SamAutomaticMaskGenerator(sam)

        print("SAM model loaded")

    def predict(self, frame: np.ndarray):
        print("Predicting...")
        masks = self.__mask_generator.generate(frame)
        img: np.ndarray = show_anns(masks)

        frame = frame.astype(np.float32)

        img = img[:, :, :3]
        img = img * 255.0

        result = 0.5 * frame + 0.5 * img
        return result