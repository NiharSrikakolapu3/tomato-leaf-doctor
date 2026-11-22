"""Where each class's advice comes from, and which page sections hold it.

Each section is a (start heading, stop heading) pair. Headings match by prefix,
ignoring case. The pages are fetched at run time, so their text stays out of the
repository (see passages.py).
"""
from dataclasses import dataclass, field

UW_PDDC = "https://pddc.wisc.edu"
CLINIC = {
    "name": "UW-Madison Plant Disease Diagnostics Clinic",
    "url": "https://pddc.wisc.edu/",
    # Contact line printed on every UW fact sheet.
    "contact": "608-262-2863 or pddc@wisc.edu",
}


@dataclass
class Source:
    title: str
    publisher: str
    url: str
    symptoms: list = field(default_factory=list)
    care: list = field(default_factory=list)
    # Start of the symptom sentence that names the sign the sheet's own photo points at.
    key_sign: str = ""


def uw_sheet(title, url, name, key_sign):
    """UW Plant Disease Facts all use the same question headings."""
    return Source(
        title=title,
        publisher="UW-Madison Division of Extension",
        url=url,
        symptoms=[(f"What does {name} look like?", "Where does")],
        care=[("How do I save", "For more information")],
        key_sign=key_sign,
    )


SOURCES = {
    "early_blight": uw_sheet(
        "Early Blight (UW Plant Disease Facts D0046)",
        f"{UW_PDDC}/2015/07/23/early-blight/",
        "early blight",
        "As these spots enlarge",
    ),
    "late_blight": uw_sheet(
        "Late Blight (UW Plant Disease Facts D0068)",
        f"{UW_PDDC}/2015/07/28/late-blight/",
        "late blight",
        "On leaves of tomato",
    ),
    "septoria_leaf_spot": uw_sheet(
        "Septoria Leaf Spot (UW Plant Disease Facts D0100)",
        f"{UW_PDDC}/2015/08/16/septoria-leaf-spot/",
        "Septoria leaf spot",
        "These spots typically",
    ),
    "bacterial_spot": uw_sheet(
        "Bacterial Spot of Tomato (UW Plant Disease Facts D0012)",
        f"{UW_PDDC}/2017/02/17/bacterial-spot-tomato/",
        "bacterial spot",
        "Bacterial spot appears on leaves",
    ),
    "mosaic_virus": uw_sheet(
        "Tobacco Mosaic (UW Plant Disease Facts D0115)",
        f"{UW_PDDC}/2021/02/19/tobacco-mosaic/",
        "tobacco mosaic",
        "On leaves, typical symptoms",
    ),
    "leaf_mold": Source(
        title="Tomato Leaf Mold (UConn Extension IPM)",
        publisher="UConn Extension",
        url="https://ipm.cahnr.uconn.edu/tomato-leaf-mold/",
        symptoms=[("Symptoms", "Host Crops"), ("Identification of Disease", "Not to Be Confused")],
        care=[("Cultural", "References")],
        key_sign="On the underside",
    ),
    "yellow_leaf_curl_virus": Source(
        title="Tomato Yellow Leaf Curl Virus (NC State Extension)",
        publisher="NC State Extension",
        url="https://content.ces.ncsu.edu/tomato-yellow-leaf-curl-virus",
        symptoms=[("Signs and Symptoms", "Look-alike conditions")],
        care=[("General Disease Management", "References")],
        key_sign="On tomatoes, the primary symptoms",
    ),
    "healthy": Source(
        title="Homegrown Tomatoes for Wisconsin (UW Extension A1691)",
        publisher="UW-Madison Division of Extension",
        url="https://hort.extension.wisc.edu/articles/home-grown-tomatoes-wisconsin/",
        symptoms=[],
        care=[("Growing season care", "Pruning"), ("Plant diseases", "Early blight")],
    ),
}
