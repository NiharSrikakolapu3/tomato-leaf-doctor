"""The whole Leaf Doctor pipeline for one photo, with every stage's output kept for the trace.

    1 leaf finder   OWLv2, frozen: best "tomato leaf" box and its score, or a retake request
    2 classifier    fine-tuned by the team, three seeds, each on the box and on the whole
                    photo: the six log-probabilities averaged
    3 gate          fitted temperature and thresholds (gate.json): advise, ask or refer
    4 advice        the checked advice table that the writer built (results/advice_table.json)

The writer runs offline over every class and pair (build_advice.py), so the gardener only
ever sees text that passed the step-to-source check or is quoted from the sheet.
"""
import io
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, Owlv2ForObjectDetection

from classifier import build_classifier
from gate import Gate, average_views
from labels import CLASSES, DISPLAY

HERE = Path(__file__).parent
FINDER_ID = "google/owlv2-base-patch16-ensemble"
FINDER_PROMPTS = ["a photo of a leaf", "a tomato leaf"]
FINDER_MIN_SCORE = 0.10
ARM_PREFIX = {"baseline": "lab", "change": "lab_field"}
SEEDS = (0, 1, 2)
CROP_MARGIN = 0.10
CACHE_SHORT_SIDE = 256
CACHE_JPEG_QUALITY = 92


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_classifier(weights, device):
    """Rebuild the network named in classifier.json beside the weights, then load them."""
    spec = json.loads((Path(weights).parent / "classifier.json").read_text())
    model, _, _, _ = build_classifier(spec["backbone"], pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(spec["mean"], spec["std"]),
    ])
    return model.to(device).eval(), transform


def as_trained(image):
    """Match the training cache: short side down to 256 and a JPEG round trip at quality 92.

    The classifier was trained and the gate fitted on crops cached this way. Skipping it
    moves the logits by about 1 and changes the route for about 1 photo in 10.
    """
    scale = CACHE_SHORT_SIDE / min(image.size)
    if scale < 1:
        size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(size, Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=CACHE_JPEG_QUALITY)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def crop_with_margin(image, box):
    x0, y0, x1, y1 = box
    pad_x = (x1 - x0) * CROP_MARGIN
    pad_y = (y1 - y0) * CROP_MARGIN
    left = max(0, int(x0 - pad_x))
    top = max(0, int(y0 - pad_y))
    right = min(image.width, int(x1 + pad_x))
    bottom = min(image.height, int(y1 + pad_y))
    if right - left < 8 or bottom - top < 8:
        return image
    return image.crop((left, top, right, bottom))


def group_key(gate, index):
    members = gate.groups[index] if gate.groups else [index]
    return "+".join(CLASSES[c] for c in members)


def group_name(gate, index):
    members = gate.groups[index] if gate.groups else [index]
    return " or ".join(DISPLAY[CLASSES[c]] for c in members)


def weight_files(folder, run):
    """Weights behind a gate run: one seed ("change_seed0") or all three ("change_ensemble")."""
    arm, which = run.split("_", 1)
    seeds = SEEDS if which == "ensemble" else [int(which.removeprefix("seed"))]
    return [Path(folder) / f"{ARM_PREFIX[arm]}_seed{seed}.pt" for seed in seeds]


class LeafDoctor:
    def __init__(self, weights, run="change_ensemble", device=None):
        """`weights` is the folder holding {lab,lab_field}_seed{0,1,2}.pt."""
        self.device = device or pick_device()
        settings = json.loads((HERE / "gate.json").read_text())
        self.views = settings["views"]
        self.gate = Gate(**settings["runs"][run])
        self.advice = json.loads((HERE / "results" / "advice_table.json").read_text())
        self.processor = AutoProcessor.from_pretrained(FINDER_ID)
        self.finder = Owlv2ForObjectDetection.from_pretrained(FINDER_ID).to(self.device).eval()
        self.classifiers = [load_classifier(f, self.device) for f in weight_files(weights, run)]
        self.backbone = json.loads((Path(weights) / "classifier.json").read_text())["backbone"]

    @torch.no_grad()
    def find_leaf(self, image):
        inputs = self.processor(text=[FINDER_PROMPTS], images=image, return_tensors="pt").to(self.device)
        outputs = self.finder(**inputs)
        size = torch.tensor([[image.height, image.width]], device=self.device)
        found = self.processor.post_process_grounded_object_detection(
            outputs, target_sizes=size, threshold=FINDER_MIN_SCORE, text_labels=[FINDER_PROMPTS]
        )[0]
        if len(found["scores"]) == 0:
            return None, 0.0
        best = int(found["scores"].argmax())
        box = [round(float(v), 1) for v in found["boxes"][best].cpu()]
        return box, round(float(found["scores"][best]), 4)

    @torch.no_grad()
    def classify(self, views):
        """Average log-probabilities of every model on every view, shape (1, 8)."""
        outputs = []
        for model, transform in self.classifiers:
            batch = torch.stack([transform(as_trained(v)) for v in views]).to(self.device)
            outputs.append(model(batch).float().cpu().numpy())
        return average_views([out[i:i + 1] for out in outputs for i in range(len(views))])

    def diagnose(self, image):
        """Run all four stages on an RGB PIL image and return the trace."""
        image = image.convert("RGB")
        box, score = self.find_leaf(image)
        crop = crop_with_margin(image, box) if box is not None else image
        logits = self.classify([crop if view == "finder_box" else image for view in self.views])

        routes, first, second, probs = self.gate.route(logits, [score])
        route = routes[0]
        top = [int(first[0]), int(second[0])]
        trace = {
            "stage1_finder": {"box": box, "score": score, "retake_below": round(self.gate.retake_below, 4)},
            "stage2_classifier": {"backbone": self.backbone, "models": len(self.classifiers),
                                  "views": self.views if box is not None else ["whole", "whole"],
                                  "log_probabilities": [round(float(v), 3) for v in logits[0]]},
            "stage3_gate": {
                "temperature": round(self.gate.temperature, 3),
                "probabilities": {group_key(self.gate, i): round(float(p), 3) for i, p in enumerate(probs[0])},
                "advise_at": round(self.gate.advise_at, 3),
                "ask_at": round(self.gate.ask_at, 3),
                "route": route,
                "candidates": [group_key(self.gate, i) for i in top],
            },
        }
        trace["stage4_advice"] = self.advice_for(route, top)
        trace["route"] = trace["stage4_advice"]["route"]
        return trace

    def advice_for(self, route, top):
        clinic = self.advice["clinic"]
        refer = {"route": "refer", "message": f"Send a sample to the {clinic['name']} "
                 f"({clinic['contact']}, {clinic['url']}).", "url": clinic["url"]}
        if route == "retake":
            return {"route": "retake", "message": "No tomato leaf was found clearly. Take a closer, "
                    "well-lit photo of one leaf, filling most of the frame."}
        if route == "advise":
            key = group_key(self.gate, top[0])
            card = self.advice["care"][key]
            return {"route": "advise", "class": key, "name": card["name"], "steps": card["steps"],
                    "warnings": card["warnings"], "sources": card["sources"]}
        if route == "ask":
            pair = [group_key(self.gate, i) for i in top]
            question = self.find_question(pair)
            if question is None or not question["asked"]:
                return refer
            return {"route": "ask", "question": question,
                    "answers": {"A": question["choices"]["A"]["leads_to"],
                                "B": question["choices"]["B"]["leads_to"], "not sure": "refer"}}
        return refer

    def find_question(self, pair):
        for question in self.advice["questions"]:
            if set(question["pair"]) == set(pair):
                return question
        return None


def open_photo(path, apply_exif=False):
    """Development photos keep their stored pixel grid, as the classifier saw them."""
    image = Image.open(path)
    if apply_exif:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def top_classes(trace, count=3):
    probabilities = trace["stage3_gate"]["probabilities"]
    return sorted(probabilities.items(), key=lambda item: -item[1])[:count]

