from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_DIR = Path(r"L:\Antigravitiy code\comfyui\Anima-Artist-Mixer")
EVIDENCE_DIR = REPO_DIR / "pr_evidence"
COMFY_OUTPUT_DIR = Path(r"I:\ComfyUI-aki-v1.6\ComfyUI\output")


def _font_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


FONT_PATH = _font_path([
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
])
BOLD_FONT_PATH = _font_path([
    Path(r"C:\Windows\Fonts\segoeuib.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
]) or FONT_PATH


def font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = BOLD_FONT_PATH if bold else FONT_PATH
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size)


TITLE = font(34, True)
H2 = font(24, True)
BODY = font(19, False)
SMALL = font(15, False)
SMALL_BOLD = font(15, True)

BG = (248, 249, 251)
CARD = (255, 255, 255)
INK = (26, 32, 44)
MUTED = (82, 92, 110)
LINE = (213, 219, 228)
GREEN = (30, 123, 86)
BLUE = (32, 91, 176)
ORANGE = (176, 101, 32)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int,
) -> int:
    x, y = xy
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if draw.textlength(candidate, font=text_font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=text_font)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=CARD, outline=LINE, width=1)
    draw.rectangle((x0, y0, x0 + 6, y1), fill=accent)
    draw.text((x0 + 18, y0 + 14), title, font=H2, fill=INK)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (240, 243, 247))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def generate_reported_issue_sheet() -> None:
    single_sheet = EVIDENCE_DIR / "compare-1024-32-single-yuchi.png"
    two_mixer = COMFY_OUTPUT_DIR / "pr4_current_preset_two_sampler_1024_32_mixer_00001_.png"
    two_prompt = COMFY_OUTPUT_DIR / "pr4_current_preset_two_sampler_1024_32_prompt_00001_.png"

    sheet = Image.new("RGB", (1500, 1120), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((36, 28), "PR #4 evidence: reported issues", font=TITLE, fill=INK)
    draw_text(
        draw,
        (36, 74),
        "Local captured runs for the multi-sampler error and the @yuchi \\(salmon-1000\\) quality report. No GitHub reply was posted.",
        BODY,
        MUTED,
        1420,
        4,
    )

    draw_card(
        draw,
        (36, 126, 1464, 500),
        "1. Same-model two-KSampler workflow: success",
        GREEN,
    )
    draw_text(
        draw,
        (62, 176),
        "Result: live 1024x1024 / 32-step two-KSampler smoke completed with status success. The patch now swaps cross_attn.forward, keeps the Attention module in place, and does not create cross_attn.original.* state-dict paths.",
        BODY,
        INK,
        650,
        4,
    )
    draw_text(
        draw,
        (62, 276),
        "Regression tests: state dict keys stay stable, full-module wrapping is proven unsafe, re-patching unwraps to the original forward, and Comfy restore leaves no .original attribute on Attention.",
        BODY,
        MUTED,
        650,
        4,
    )
    sheet.paste(fit_image(two_prompt, (330, 330)), (770, 154))
    sheet.paste(fit_image(two_mixer, (330, 330)), (1110, 154))
    draw.text((778, 466), "bypass branch: prompt/no mixer", font=SMALL_BOLD, fill=INK)
    draw.text((1118, 466), "patched branch: balanced mixer", font=SMALL_BOLD, fill=INK)

    draw_card(
        draw,
        (36, 532, 1464, 900),
        "2. @yuchi \\(salmon-1000\\): no mixer regression",
        BLUE,
    )
    draw_text(
        draw,
        (62, 582),
        "Same seed, model, prompt, sampler, scheduler, and no preset. Current mixer and original mixer have identical recorded distances to the direct prompt, so the difference is not a regression introduced by this PR.",
        BODY,
        INK,
        600,
        4,
    )
    draw_text(
        draw,
        (62, 690),
        "Direct prompt conditioning and mixer conditioning are expected to differ: one encodes the artist in the full LLM prompt; the other encodes artist/base separately and mixes at cross-attention output.",
        BODY,
        MUTED,
        600,
        4,
    )
    metrics = [
        ("current vs prompt descriptor", "0.171010"),
        ("original vs prompt descriptor", "0.171010"),
        ("current mixer time", "41.8s"),
        ("original mixer time", "43.2s"),
    ]
    y = 796
    for label, value in metrics:
        draw.text((62, y), label, font=SMALL, fill=MUTED)
        draw.text((330, y), value, font=SMALL_BOLD, fill=INK)
        y += 24

    sheet.paste(fit_image(single_sheet, (740, 320)), (700, 560))
    draw.text((708, 884), "direct prompt / original mixer / current mixer", font=SMALL_BOLD, fill=INK)

    draw_card(draw, (36, 932, 1464, 1084), "Practical value shown by the captured runs", ORANGE)
    draw_text(
        draw,
        (62, 984),
        "The PR keeps the original balanced behavior for regression safety, fixes same-model branch workflows, and adds safer presets, layer/timing controls, recipe/probe tooling, and tests. Fresh 1/2/4-artist evidence is saved as fresh-multi-artist-matrix.png.",
        BODY,
        INK,
        1360,
        4,
    )
    sheet.save(EVIDENCE_DIR / "compare-pr4-reported-issues.png")


def generate_mode_summary_sheet() -> None:
    fresh_sheet = EVIDENCE_DIR / "fresh-multi-artist-matrix.png"
    summary = Image.new("RGB", (1500, 1280), BG)
    draw = ImageDraw.Draw(summary)
    draw.text((36, 28), "Mode and preset value summary", font=TITLE, fill=INK)
    draw_text(
        draw,
        (36, 74),
        "Clear function statement for the modes and presets added or kept in PR #4, with timing evidence from local captured runs.",
        BODY,
        MUTED,
        1420,
        4,
    )

    draw_card(draw, (36, 126, 710, 430), "Core combine/fusion modes", BLUE)
    core_rows = [
        ("output_avg", "quality default; averages per-artist attention outputs"),
        ("concat", "fast/compatibility path; concatenates contexts"),
        ("lowrank_avg", "deterministic low-rank artist-delta stability"),
        ("interpolate", "smooth base-to-artist strength control"),
        ("base_preserve", "content-preserving style direction"),
        ("concat_with_base", "fast base+artist context path"),
    ]
    y = 180
    for name, desc in core_rows:
        draw.text((64, y), name, font=SMALL_BOLD, fill=INK)
        draw_text(draw, (210, y), desc, SMALL, MUTED, 450, 2)
        y += 38

    draw_card(draw, (740, 126, 1464, 430), "Reported issue timing", GREEN)
    time_rows = [
        ("fresh single prompt/balanced", "3.1s / 3.0s"),
        ("fresh double prompt/balanced", "3.4s / 4.1s"),
        ("fresh four-artist prompt/balanced", "3.2s / 5.0s"),
        ("fresh four-artist drift_auto", "4.0s"),
        ("two-KSampler same-model smoke", "success"),
    ]
    y = 180
    for name, value in time_rows:
        draw.text((770, y), name, font=SMALL, fill=MUTED)
        draw.text((1190, y), value, font=SMALL_BOLD, fill=INK)
        y += 42
    draw_text(
        draw,
        (770, 388),
        "Timings are cost visibility only. They are not a speedup claim.",
        SMALL,
        MUTED,
        650,
        2,
    )

    draw_card(draw, (36, 462, 1464, 812), "Preset functions", ORANGE)
    preset_rows = [
        ("prompt_passthrough", "direct prompt/no-mixer parity with positive weights"),
        ("balanced", "default original-style mixer: output_avg + interpolate, stabilizers off"),
        ("strong_style", "higher style strength with light EMA"),
        ("stable_seed", "static capture for lower cross-seed style drift"),
        ("drift_soft", "softer low-drift portrait/fullbody route"),
        ("face_lock", "base_preserve + token norm lock for face-focused prompts"),
        ("scene_lock", "base_preserve route for wide/background-heavy prompts"),
        ("drift_auto", "runtime router by prompt shape and artist count"),
        ("anchor_lock", "strongest anchor-Q style consistency path"),
        ("fast_preview", "quick concat preview path"),
        ("identity_guard", "conservative low-rank/base-preserve identity path"),
        ("compatibility_safe", "concat path with stabilizers disabled for other patch nodes"),
    ]
    for idx, (name, desc) in enumerate(preset_rows):
        if idx < 6:
            x = 64
            yy = 514 + idx * 48
        else:
            x = 760
            yy = 514 + (idx - 6) * 48
        draw.text((x, yy), name, font=SMALL_BOLD, fill=INK)
        draw_text(draw, (x + 170, yy), desc, SMALL, MUTED, 500, 2)

    draw_card(draw, (36, 844, 1464, 1238), "Image evidence index", BLUE)
    summary.paste(fit_image(fresh_sheet, (660, 330)), (64, 890))
    draw.text((72, 1220), "fresh 1 / 2 / 4 artist prompt vs balanced vs drift_auto", font=SMALL_BOLD, fill=INK)
    index_text = [
        ("compare-pr4-reported-issues.png", "answers the two reviewer reports visually"),
        ("fresh-multi-artist-matrix.png", "fresh 1/2/4 artist value evidence"),
        ("compare-1024-32-single-yuchi.png", "@yuchi direct/original/current comparison"),
        ("compare-1024-32-preset-two-sampler-yuchi.png", "two-KSampler same-model success evidence"),
        ("mode_preset_matrix.md", "reviewer-readable mode and preset function table"),
        ("pr4_reported_issues.metrics.json", "machine-readable timings and distances"),
    ]
    y = 900
    for file_name, desc in index_text:
        draw.text((770, y), file_name, font=SMALL_BOLD, fill=INK)
        draw_text(draw, (770, y + 22), desc, SMALL, MUTED, 620, 2)
        y += 68
    summary.save(EVIDENCE_DIR / "mode-preset-value-summary.png")


def generate_two_sampler_sheet() -> None:
    two_mixer = COMFY_OUTPUT_DIR / "pr4_current_preset_two_sampler_1024_32_mixer_00001_.png"
    two_prompt = COMFY_OUTPUT_DIR / "pr4_current_preset_two_sampler_1024_32_prompt_00001_.png"
    contact = Image.new("RGB", (1040, 610), (255, 255, 255))
    draw = ImageDraw.Draw(contact)
    draw.text((18, 14), "Two-KSampler same-model smoke: success", font=H2, fill=INK)
    draw_text(
        draw,
        (18, 48),
        "1024x1024, 32 steps, balanced preset. One branch bypasses the mixer, one branch uses AnimaArtistCrossAttn. Both images were produced in the same workflow without the reported .original error.",
        SMALL,
        MUTED,
        1000,
        2,
    )
    contact.paste(fit_image(two_prompt, (500, 500)), (18, 92))
    contact.paste(fit_image(two_mixer, (500, 500)), (522, 92))
    draw.text((28, 570), "prompt / no mixer branch", font=SMALL_BOLD, fill=INK)
    draw.text((532, 570), "balanced mixer branch", font=SMALL_BOLD, fill=INK)
    contact.save(EVIDENCE_DIR / "compare-1024-32-preset-two-sampler-yuchi.png")


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    generate_two_sampler_sheet()
    generate_reported_issue_sheet()
    generate_mode_summary_sheet()


if __name__ == "__main__":
    main()
