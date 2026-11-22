"""Class order shared with the classifier, and the names a gardener reads."""

# Same order as the classifier's 8 outputs.
CLASSES = [
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "mosaic_virus",
    "yellow_leaf_curl_virus",
    "healthy",
]

DISPLAY = {
    "bacterial_spot": "bacterial spot",
    "early_blight": "early blight",
    "late_blight": "late blight",
    "leaf_mold": "leaf mold",
    "septoria_leaf_spot": "Septoria leaf spot",
    "mosaic_virus": "mosaic virus",
    "yellow_leaf_curl_virus": "tomato yellow leaf curl virus",
    "healthy": "no disease (healthy)",
}

DISEASES = [c for c in CLASSES if c != "healthy"]
