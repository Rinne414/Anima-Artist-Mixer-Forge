"""Chain building, previewing, and report formatting helpers."""

from .constants import (
    CHAIN_LAYOUT_EVEN_LAYERS,
    CHAIN_LAYOUT_LAYER_SCHEDULED,
    CHAIN_LAYOUT_MANUAL,
    MAX_ARTISTS,
    WEIGHT_MAX,
    WEIGHT_MIN,
)
from .options import format_bool
from .parsing import (
    clamp_float,
    parse_artist_layer_routes,
    parse_artist_timing_routes,
    parse_artist_weights,
    parse_layer_filter,
    parse_timing_filter,
    resolve_artist_layer_routes,
    resolve_artist_timing_routes,
    split_artist_chain,
)


def format_route_timing(label, timing):
    if timing is None:
        return label
    start, end, fade = timing
    text = f"{label}%{start:.2f}-{end:.2f}"
    if fade > 0.0:
        text += f"~{fade:.2g}"
    return text


def format_layer_span(start, end):
    return f"L{start}" if start == end else f"L{start}-L{end}"


def format_artist_block_map(labels, layer_route_texts, timing_texts,
                            num_blocks, target_blocks=None):
    labels = list(labels or [])
    if not labels:
        return "  (none)"
    if target_blocks is None:
        target_blocks = list(range(int(num_blocks)))
    else:
        target_blocks = list(target_blocks)
    if not target_blocks:
        return "  (no patched blocks)"
    layer_routes, _ = resolve_artist_layer_routes(layer_route_texts or [], num_blocks)
    timing_routes, _ = resolve_artist_timing_routes(timing_texts or [])
    if len(layer_routes) < len(labels):
        layer_routes.extend([None] * (len(labels) - len(layer_routes)))
    if len(timing_routes) < len(labels):
        timing_routes.extend([None] * (len(labels) - len(timing_routes)))

    rows = []
    for block_idx in target_blocks:
        active = []
        for label, route, timing in zip(labels, layer_routes, timing_routes):
            if route is None or block_idx in route:
                active.append(format_route_timing(str(label), timing))
        rows.append((block_idx, tuple(active)))

    grouped = []
    start = end = rows[0][0]
    prev_active = rows[0][1]
    for block_idx, active in rows[1:]:
        if active == prev_active and block_idx == end + 1:
            end = block_idx
            continue
        grouped.append((start, end, prev_active))
        start = end = block_idx
        prev_active = active
    grouped.append((start, end, prev_active))

    lines = []
    for start, end, active in grouped:
        names = ", ".join(active) if active else "(original cross-attn)"
        lines.append(f"  {format_layer_span(start, end)}: {names}")
    return "\n".join(lines)


def sanitize_artist_name_for_builder(name):
    return str(name or "").strip()


def format_weighted_artist_name(name, weight):
    weight = clamp_float(weight, WEIGHT_MIN, WEIGHT_MAX)
    if abs(weight - 1.0) <= 1e-6:
        return name
    return f"::{name}::{weight:.3g}"


def _format_route_float(value):
    return f"{float(value):.2f}"


def _even_layer_routes(count, num_blocks):
    routes = []
    for idx in range(count):
        lo = int(round(idx * num_blocks / count))
        hi = int(round((idx + 1) * num_blocks / count)) - 1
        routes.append(f"{lo}-{max(lo, hi)}")
    return routes


def default_builder_routes(layout, count, num_blocks):
    count = max(0, int(count))
    if count <= 0:
        return [], []
    num_blocks = max(1, int(num_blocks))
    if layout == CHAIN_LAYOUT_EVEN_LAYERS:
        return _even_layer_routes(count, num_blocks), [""] * count
    if layout == CHAIN_LAYOUT_LAYER_SCHEDULED:
        if count <= 3:
            route_templates = ["0-8", "9-18", "19-27"]
            timing_templates = ["0.0-0.45", "0.35-0.85", "0.65-1.0"]
            routes = []
            timings = []
            for idx in range(count):
                parsed = parse_layer_filter(route_templates[idx], num_blocks)
                if parsed is None:
                    routes.append("")
                else:
                    routes.append(f"{parsed[0]}-{parsed[-1]}")
                timings.append(timing_templates[idx])
            return routes, timings
        routes = []
        timings = []
        for idx in range(count):
            lo = int(round(idx * num_blocks / count))
            hi = int(round((idx + 1) * num_blocks / count)) - 1
            routes.append(f"{lo}-{max(lo, hi)}")
            start = max(0.0, (idx / count) - 0.08)
            end = min(1.0, ((idx + 1) / count) + 0.08)
            timings.append(f"{_format_route_float(start)}-{_format_route_float(end)}")
        return routes, timings
    return [""] * count, [""] * count


def parse_builder_artist_table(artist_table, return_warnings=False):
    rows = []
    warnings = []
    for raw_line in str(artist_table or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        name = parts[0] if parts else ""
        weight = 1.0
        if len(parts) > 1 and parts[1]:
            try:
                weight = float(parts[1])
            except ValueError:
                label = name or "(empty artist)"
                warnings.append(f"invalid weight for {label}: {parts[1]}; using 1.0")
                weight = 1.0
        layer_route = parts[2] if len(parts) > 2 else ""
        timing_route = parts[3] if len(parts) > 3 else ""
        rows.append((name, weight, layer_route, timing_route))
    if return_warnings:
        return rows, warnings
    return rows


def build_artist_chain_from_rows(layout, rows, num_blocks=28, extra_warnings=None):
    cleaned_rows = []
    for name, weight, layer_route, timing_route in rows:
        name = sanitize_artist_name_for_builder(name)
        if not name:
            continue
        cleaned_rows.append((name, float(weight), str(layer_route or "").strip(),
                             str(timing_route or "").strip()))
    routes, timings = default_builder_routes(layout, len(cleaned_rows), num_blocks)
    entries = []
    labels = []
    weights = []
    layer_routes = []
    timing_routes = []
    warnings = list(extra_warnings or [])
    if not cleaned_rows:
        warnings.append("no artists; add at least one artist")
    for idx, (name, weight, layer_route, timing_route) in enumerate(cleaned_rows):
        if layout != CHAIN_LAYOUT_MANUAL:
            if not layer_route and idx < len(routes):
                layer_route = routes[idx]
            if not timing_route and idx < len(timings):
                timing_route = timings[idx]
        if layer_route and parse_layer_filter(layer_route, num_blocks) is None:
            warnings.append(f"invalid layer route ignored for {name}: {layer_route}")
            layer_route = ""
        if timing_route and parse_timing_filter(timing_route) is None:
            warnings.append(f"invalid timing route ignored for {name}: {timing_route}")
            timing_route = ""
        entry = format_weighted_artist_name(name, weight)
        if layer_route:
            entry = f"{entry}@{layer_route}"
        if timing_route:
            entry = f"{entry}%{timing_route}"
        entries.append(entry)
        labels.append(name)
        weights.append(weight)
        layer_routes.append(layer_route)
        timing_routes.append(timing_route)

    chain = "\n".join(entries)
    status = "CHECK" if warnings else "OK"
    lines = [
        "Anima Artist Chain Builder",
        "",
        f"status: {status}",
        f"layout: {layout}",
        f"artists: {len(entries)}",
        "",
        "artist_chain:",
        chain or "  (empty)",
        "",
        "block map:",
        format_artist_block_map(labels, layer_routes, timing_routes, num_blocks),
        "",
        "warnings:",
    ]
    if warnings:
        lines.extend(f"  - {w}" for w in warnings)
    else:
        lines.append("  - no obvious builder issue")
    lines.extend([
        "",
        "next steps:",
        "  - connect artist_chain to AnimaArtistPack.artist_chain",
        "  - connect AnimaArtistPack.artist_pack to AnimaArtistPresetApply.artist_pack",
        "  - use AnimaArtistPreset first; switch to compatibility_safe if other attention patch nodes are present",
    ])
    return chain, "\n".join(lines)


def format_artist_chain_preview(artist_chain, num_blocks=28):
    parts = split_artist_chain(artist_chain)
    clean_timing_parts, timing_routes = parse_artist_timing_routes(parts)
    clean_layer_parts, layer_routes = parse_artist_layer_routes(clean_timing_parts)
    names, weights, has_explicit = parse_artist_weights(clean_layer_parts)

    warnings = []
    for raw, clean, timing in zip(parts, clean_timing_parts, timing_routes):
        if "%" in str(raw) and not timing:
            warnings.append(f"invalid timing route kept as artist text: {raw}")
    for raw, clean, route in zip(clean_timing_parts, clean_layer_parts, layer_routes):
        if "@" in str(raw) and not route:
            suffix = str(raw).rsplit("@", 1)[-1].strip()
            if not any(ch.isdigit() or ch in ",，-" for ch in suffix):
                continue
            warnings.append(f"invalid layer route kept as artist text: {raw}")
    if len(names) > MAX_ARTISTS:
        warnings.append(f"artist count {len(names)} exceeds MAX_ARTISTS={MAX_ARTISTS}; Pack will truncate")
    if has_explicit:
        warnings.append("::weight detected; runtime normalize_weights will be bypassed")
    if any(w < 0.0 for w in weights):
        warnings.append("negative ::weight detected; those artists subtract style instead of adding it")

    cleaned_entries = []
    for label, weight, layer_route, timing_route in zip(names, weights, layer_routes, timing_routes):
        entry = format_weighted_artist_name(label, weight)
        if layer_route:
            entry = f"{entry}@{layer_route}"
        if timing_route:
            entry = f"{entry}%{timing_route}"
        cleaned_entries.append(entry)
    cleaned_chain = "\n".join(cleaned_entries)
    status = "CHECK" if warnings else "OK"

    lines = [
        "Anima Artist Chain Preview",
        "",
        f"status: {status}",
        f"artists: {len(names)}",
        f"explicit weights: {format_bool(has_explicit)}",
        "",
        "parsed artists:",
    ]
    if names:
        for idx, (label, weight, layer_route, timing_route) in enumerate(
            zip(names, weights, layer_routes, timing_routes), start=1,
        ):
            layer_text = f" @ {layer_route}" if layer_route else ""
            timing_text = f" % {timing_route}" if timing_route else ""
            lines.append(f"  {idx}. {label} :: {weight:.3g}{layer_text}{timing_text}")
    else:
        lines.append("  (none)")
    lines.extend([
        "",
        "block map:",
        format_artist_block_map(names, layer_routes, timing_routes, num_blocks),
        "",
        "warnings:",
    ])
    if warnings:
        lines.extend(f"  - {w}" for w in warnings)
    else:
        lines.append("  - no obvious syntax issue")
    lines.extend([
        "",
        "next steps:",
        "  - if this report looks correct, connect cleaned_chain to AnimaArtistPack.artist_chain",
        "  - use AnimaArtistInspector after Pack to verify effective weights and routing",
    ])
    return cleaned_chain, "\n".join(lines)
