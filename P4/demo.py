"""Gradio demo: upload a tomato leaf photo, see each stage, and get the triage decision.

Usage:
    python demo.py --weights DIR [--examples DIR] [--share]
In Colab, --share prints a public link.
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import LeafDoctor, top_classes

ROUTE_TITLES = {
    "advise": "Advice",
    "ask": "One question first",
    "refer": "Ask the clinic",
    "retake": "Please retake the photo",
}


def draw_box(image, box):
    shown = image.copy()
    if box:
        ImageDraw.Draw(shown).rectangle(box, outline=(255, 64, 32), width=max(3, image.width // 150))
    return shown


def care_markdown(card):
    lines = [f"### Likely: {card['name']}", "", "**Care steps**", ""]
    for number, step in enumerate(card["steps"], 1):
        note = " *(quoted from the fact sheet)*" if step["quoted"] else ""
        lines.append(f"{number}. {step['shown']}{note}")
    if card["warnings"]:
        lines += ["", "**Warnings from the fact sheet**", ""]
        lines += [f"- {w['shown']}" for w in card["warnings"]]
    lines += ["", "Sources: " + ", ".join(f"[{s['title']}]({s['url']})" for s in card["sources"])]
    return "\n".join(lines)


def advice_markdown(doctor, advice):
    route = advice["route"]
    if route == "advise":
        return care_markdown(advice)
    if route == "ask":
        question = advice["question"]
        a = question["choices"]["A"]["shown"]
        b = question["choices"]["B"]["shown"]
        return f"### {question['stem']}\n\n**A.** {a}\n\n**B.** {b}\n\nPick an answer below."
    return f"### {ROUTE_TITLES[route]}\n\n{advice['message']}"


def build(doctor, examples):
    def run(image):
        if image is None:
            return None, {}, "Upload a photo first.", None, gr.update(visible=False, value=None)
        trace = doctor.diagnose(image)
        box = trace["stage1_finder"]["box"]
        scores = dict(top_classes(trace, 3))
        text = advice_markdown(doctor, trace["stage4_advice"])
        asking = trace["route"] == "ask"
        return draw_box(image, box), scores, text, trace, gr.update(visible=asking, value=None)

    def answer(choice, trace):
        if not trace or not choice:
            return ""
        answers = trace["stage4_advice"]["answers"]
        key = answers.get(choice, "refer")
        if key == "refer":
            return advice_markdown(doctor, doctor.advice_for("refer", []))
        card = doctor.advice["care"][key]
        return care_markdown({**card, "name": card["name"]})

    with gr.Blocks(title="Leaf Doctor") as app:
        gr.Markdown("## Leaf Doctor\nA tomato leaf triage assistant. Upload a photo of one leaf.")
        trace_state = gr.State()
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="pil", label="Photo")
                go = gr.Button("Check this leaf", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=photo)
            with gr.Column():
                found = gr.Image(label="Stage 1: leaf the finder picked", interactive=False)
                scores = gr.Label(label="Stage 2 and 3: calibrated probabilities", num_top_classes=3)
        advice = gr.Markdown()
        choice = gr.Radio(["A", "B", "not sure"], label="Your answer", visible=False)
        follow_up = gr.Markdown()
        with gr.Accordion("Full trace", open=False):
            trace_view = gr.JSON()

        go.click(run, inputs=photo, outputs=[found, scores, advice, trace_state, choice])
        trace_state.change(lambda t: t, inputs=trace_state, outputs=trace_view)
        choice.change(answer, inputs=[choice, trace_state], outputs=follow_up)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run", default="change_ensemble")
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    examples = sorted(str(p) for p in args.examples.glob("*.jpg")) if args.examples else []
    doctor = LeafDoctor(args.weights, args.run)
    build(doctor, examples).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
