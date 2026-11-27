import torch
from PIL import Image
from transformers import  OwlViTForObjectDetection, OwlViTProcessor

model_name = "google/owlvit-base-patch32"
threshold = 0.1

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_model():
    # load model
    model = OwlViTForObjectDetection.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # load processor
    processor = OwlViTProcessor.from_pretrained(model_name)

    return model, processor


def detect(model, processor, labels, image, threshold=threshold):
    inputs = processor(text = [labels], return_tensors="pt", images=image, padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(outputs, threshold, target_sizes=[(image.height, image.width)], text_labels=[labels])
    result = results[0]
    
    boxes = result["boxes"]
    scores = result["scores"]
    text_labels = result["text_labels"]

    boxes = boxes.cpu().tolist()
    scores = scores.cpu().tolist()

    return boxes, scores, text_labels

if __name__ == "__main__":

