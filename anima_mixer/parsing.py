"""Artist chain parsing: splitting, weights, layer routes, timing routes.

Chain syntax overview (all parts optional, composable):

    wlop                          plain artist, weight 1.0
    ::wlop::1.5                   linear injection weight 1.5
    ::wlop::-0.5                  negative weight = style subtraction
    (wlop:1.1)                    CLIP-side weighting (non-linear), kept verbatim
    ::(wlop:1.1)::0.8             both stacked
    wlop@0-8                      layer route: inject only into DiT blocks 0-8
    wlop%0.0-0.45                 timing route: active only for sampling progress 0.0-0.45
    wlop%0.0-0.45~0.1             timing route with smoothstep fade width 0.1
    1.2::wlop@0-8%0.0-0.45~0.1::  everything combined
"""

from .constants import WEIGHT_MAX, WEIGHT_MIN


def clamp_float(value, lo, hi):
    value = float(value)
    if value != value:  # NaN compares unequal to itself
        raise ValueError("NaN is not a valid value")
    return max(lo, min(hi, value))


def _is_float(text):
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def strip_prefix_weight_boundary(text):
    s = str(text or "").strip()
    if not s.endswith("::") or "::" not in s[:-2]:
        return s
    head, _, tail = s[:-2].partition("::")
    if not tail.strip():
        return s
    try:
        float(head.strip())
    except ValueError:
        return s
    return s[:-2].strip()


def _is_timing_suffix_text(text):
    s = str(text or "").strip()
    return bool(s) and all(ch in set("0123456789.-~ ") for ch in s)


def _is_layer_route_segment(text):
    s = str(text or "").strip()
    if not s:
        return False
    if "%" in s:
        layer_text, timing = s.rsplit("%", 1)
        if _is_timing_suffix_text(timing):
            s = layer_text
    return bool(s.strip()) and all(ch in set("0123456789-,， .%") for ch in s)


def _comma_continues_layer_route(current_text, next_text):
    cur = str(current_text or "")
    at_idx = cur.rfind("@")
    if at_idx < 0:
        return False
    tail = cur[at_idx + 1:]
    if "%" in tail:
        layer_text, timing = tail.rsplit("%", 1)
        if _is_timing_suffix_text(timing):
            tail = layer_text
    if not _is_layer_route_segment(tail):
        return False
    # The next comma-separated piece may carry the entry's closing marker or a
    # legacy postfix weight (e.g. '1.2::wlop@0,2,4::' or 'wlop@0,2,4::1.2').
    # Partition on the FIRST '::': the closing tail must be empty or a float
    # for this to still be one route; the head is the real route segment.
    nxt = str(next_text or "").strip()
    if "::" in nxt:
        head, _, closing = nxt.partition("::")
        closing = closing.strip()
        if closing and not _is_float(closing):
            return False
        nxt = head.strip()
    return _is_layer_route_segment(nxt)


def split_artist_chain(chain):
    """Split an artist chain. Commas separate artists but may also appear
    inside a trailing @layer route (e.g. ``wlop@0,2,4``)."""
    if not chain:
        return []
    s = str(chain).replace("\r", "\n")
    parts = []
    buf = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\n":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            i += 1
            continue
        if ch in ",，":
            j = i + 1
            while j < len(s) and s[j] not in ",，\n":
                j += 1
            next_piece = s[i + 1:j]
            if _comma_continues_layer_route("".join(buf), next_piece):
                buf.append(ch)
            else:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


def _strip_colon_markers(name):
    """Clean leftover ``::`` markers from a name after weight parsing.

    Handles the fully decorative form ``::text::`` (both sides, no weight), a
    trailing ``::<float>`` remnant left when a prefix weight already won a
    double-weight entry like ``1.5::wlop::0.8``, and a bare trailing ``::``
    (empty weight) left when a route detached from ``wlop@0-8::``.
    """
    s = str(name).strip()
    if s.startswith("::") and s.endswith("::"):
        inner = s[2:-2].strip()
        if inner and "::" not in inner:
            return inner
    if not s.startswith("::") and s.endswith("::") and "::" not in s[:-2]:
        head = s[:-2].strip()
        if head:
            return head
    if not s.startswith("::") and "::" in s:
        name_part, _, w_part = s.rpartition("::")
        w_part = w_part.strip()
        if name_part.strip() and w_part and _is_float(w_part):
            return name_part.strip()
    return s


def parse_artist_entries(parts):
    """Parse each artist entry into ``(name, weight, explicit)``.

    Prefix syntax (v26 recommended, NovelAI-style):
      'weight::name'        e.g. '1.5::wlop'
      'weight::(name:w)'    e.g. '0.8::(wlop:1.1)'  # with parentheses

    Postfix syntax (v24-v25 compatible):
      'name::weight'        e.g. 'wlop::1.5'
      '::name::weight'      e.g. '::wlop::1.5'

    Examples:
      'wlop'              → ('wlop', 1.0, False)
      '1.5::wlop'         → ('wlop', 1.5, True)              # v26 prefix (recommended)
      'wlop::1.5'         → ('wlop', 1.5, True)              # postfix (compatible)
      '::wlop::1.5'       → ('wlop', 1.5, True)              # postfix (compatible)
      '(wlop:1.1)'        → ('(wlop:1.1)', 1.0, False)       # CLIP parentheses only
      '0.8::(wlop:1.1)'   → ('(wlop:1.1)', 0.8, True)        # prefix + parentheses
      '(wlop:1.1)::0.8'   → ('(wlop:1.1)', 0.8, True)        # postfix + parentheses
      '::wlop'            → ('wlop', 1.0, False)             # decorative prefix
      '::sakimichan::'    → ('sakimichan', 1.0, False)       # decorative both sides
      '1.5::wlop::0.8'    → ('wlop', 1.5, True)              # double weight: prefix wins

    Parsing order: try prefix first (::before is valid number), fall back to
    postfix (::after is valid number). Weights are clamped to
    [WEIGHT_MIN, WEIGHT_MAX]; negative weights subtract the artist's style.
    NaN weights are treated as invalid input and the raw text is kept.
    """
    entries = []
    for raw in parts:
        s = str(raw).strip()
        if not s:
            continue
        weight = 1.0
        explicit = False

        if "::" in s:
            # ── Try prefix syntax first: 'weight::name' ──
            head, _, tail = s.partition("::")
            head_stripped = head.strip()
            tail_stripped = tail.strip()
            if tail_stripped.endswith("::"):
                tail_stripped = tail_stripped[:-2].strip()
            if head_stripped and tail_stripped:
                try:
                    weight = clamp_float(head_stripped, WEIGHT_MIN, WEIGHT_MAX)
                    explicit = True
                    s = tail_stripped
                except ValueError:
                    pass

            # ── If prefix didn't match, try postfix: 'name::weight' or '::name::weight' ──
            if not explicit:
                tmp = s[2:] if s.startswith("::") else s
                if "::" in tmp:
                    name_part, _, w_part = tmp.rpartition("::")
                    try:
                        weight = clamp_float(w_part.strip(), WEIGHT_MIN, WEIGHT_MAX)
                        explicit = True
                        s = name_part.strip()
                    except ValueError:
                        # Neither prefix nor postfix is valid number → keep raw text
                        pass
                elif s.startswith("::"):
                    # Decorative prefix '::wlop' → strip prefix, weight defaults to 1.0
                    s = tmp.strip()

            # ── Clean decorative '::text::' and double-weight remnants ──
            s = _strip_colon_markers(s)

        if not s:
            continue
        entries.append((s, weight, explicit))
    return entries


def parse_artist_weights(parts):
    """Thin wrapper over :func:`parse_artist_entries`.

    Returns ``(names, weights, has_explicit)`` where ``has_explicit`` is True
    when at least one artist specified an explicit weight.
    """
    names = []
    weights = []
    has_explicit = False
    for name, weight, explicit in parse_artist_entries(parts):
        names.append(name)
        weights.append(weight)
        if explicit:
            has_explicit = True
    return names, weights, has_explicit


def _detach_trailing_postfix_weight(base, route):
    """Move a trailing ``::<weight>`` off a route back onto the base.

    Postfix-weight forms like ``wlop@0,2,4::1.2`` split into base ``wlop`` and
    route ``0,2,4::1.2``. The route charset rejects ``:``, so pull the closing
    ``::<float>`` (or bare ``::``) back onto the base for the weight parser and
    return the clean route segment.
    """
    if "::" not in route:
        return base, route
    core, _, suffix = route.partition("::")
    suffix = suffix.strip()
    if suffix and not _is_float(suffix):
        return base, route
    sep = f"::{suffix}" if suffix else "::"
    return f"{base}{sep}", core


def parse_artist_layer_route(name):
    """Parse a trailing ``@layer_filter`` route from a single artist entry.

    Only the final ``@`` is considered, so artist tags that legitimately
    start with ``@`` (e.g. ``@wlop``) survive untouched.
    """
    s = strip_prefix_weight_boundary(name)
    if not s or "@" not in s:
        return s, ""
    base, route = s.rsplit("@", 1)
    route = route.strip()
    if not route:
        return s, ""
    base, route = _detach_trailing_postfix_weight(base, route)
    allowed = set("0123456789,- ，.%")
    if route and all(ch in allowed for ch in route):
        base = base.strip()
        if base:
            return base, route
    return s, ""


def parse_artist_layer_routes(names):
    clean_names = []
    routes = []
    for name in names:
        clean, route = parse_artist_layer_route(name)
        clean_names.append(clean)
        routes.append(route)
    return clean_names, routes


def parse_artist_timing_route(name):
    """Parse a trailing ``%start-end`` or ``%start-end~fade`` timing route.

    Like layer routes, only the final ``%`` is considered; invalid suffixes
    are kept as plain artist text.
    """
    s = strip_prefix_weight_boundary(name)
    if not s or "%" not in s:
        return s, ""
    base, timing = s.rsplit("%", 1)
    timing = timing.strip()
    if not timing:
        return s, ""
    base, timing = _detach_trailing_postfix_weight(base, timing)
    allowed = set("0123456789.-~ ")
    if timing and all(ch in allowed for ch in timing):
        if parse_timing_filter(timing) is None:
            return s, ""
        base = base.strip()
        if base:
            return base, timing
    return s, ""


def parse_artist_timing_routes(names):
    clean_names = []
    timings = []
    for name in names:
        clean, timing = parse_artist_timing_route(name)
        clean_names.append(clean)
        timings.append(timing)
    return clean_names, timings


def _parse_layer_percent_value(text, has_percent_marker):
    s = str(text or "").strip()
    if not s:
        return None
    has_suffix = s.endswith("%")
    if has_suffix:
        s = s[:-1].strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    if value != value:  # NaN
        return None
    if has_suffix:
        value /= 100.0
    elif has_percent_marker and abs(value) > 1.0:
        value /= 100.0
    if value < 0.0 or value > 1.0:
        return None
    return float(value)


def _layer_percent_point_to_block(percent, num_blocks):
    if num_blocks <= 1:
        return 0
    return int(round(clamp_float(percent, 0.0, 1.0) * (num_blocks - 1)))


def _layer_percent_window_to_blocks(start, end, num_blocks):
    if num_blocks <= 1:
        return 0, 0
    lo_percent = clamp_float(start, 0.0, 1.0)
    hi_percent = clamp_float(end, 0.0, 1.0)
    if lo_percent > hi_percent:
        lo_percent, hi_percent = hi_percent, lo_percent
    if abs(hi_percent - lo_percent) <= 1e-8:
        idx = _layer_percent_point_to_block(lo_percent, num_blocks)
        return idx, idx
    lo = int(round(lo_percent * num_blocks))
    hi = int(round(hi_percent * num_blocks)) - 1
    lo = max(0, min(num_blocks - 1, lo))
    hi = max(0, min(num_blocks - 1, hi))
    if hi < lo:
        hi = lo
    return lo, hi


def _parse_layer_percent_part(part, num_blocks):
    has_percent_marker = "%" in str(part or "")
    if not has_percent_marker and "." not in str(part or ""):
        return None
    if "-" in str(part or "")[1:]:
        dash_idx = str(part).index("-", 1)
        start = _parse_layer_percent_value(part[:dash_idx], has_percent_marker)
        end = _parse_layer_percent_value(part[dash_idx + 1:], has_percent_marker)
        if start is None or end is None:
            return None
        lo, hi = _layer_percent_window_to_blocks(start, end, num_blocks)
        return list(range(lo, hi + 1))
    percent = _parse_layer_percent_value(part, has_percent_marker)
    if percent is None:
        return None
    return [_layer_percent_point_to_block(percent, num_blocks)]


def parse_layer_filter(text, num_blocks, warnings=None):
    """Parse a layer filter like ``0,3,5-10,-1`` into a sorted block list.

    Returns:
      None  when ``text`` is empty or contains nothing recognizable,
      []    when at least one numeric part was recognized but the result
            matches no valid block (e.g. ``30-40`` on a 28-block model),
      [..]  the sorted set of matched blocks otherwise.

    When ``warnings`` is a list, an unparseable part or an empty-but-recognized
    result appends an explanatory message instead of being swallowed silently.
    """
    if not text:
        return None
    s = str(text).replace("，", ",").replace(" ", "")
    if not s:
        return None
    result = set()
    recognized = False
    for part in s.split(","):
        if not part:
            continue
        percent_blocks = _parse_layer_percent_part(part, num_blocks)
        if percent_blocks is not None:
            recognized = True
            result.update(percent_blocks)
            continue
        if "-" in part[1:]:
            dash_idx = part.index("-", 1)
            try:
                lo = int(part[:dash_idx])
                hi = int(part[dash_idx + 1:])
            except ValueError:
                if warnings is not None:
                    warnings.append(f"layer route part {part!r} ignored")
                continue
            recognized = True
            if lo < 0:
                lo += num_blocks
            if hi < 0:
                hi += num_blocks
            if lo > hi:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(num_blocks - 1, hi)
            if lo <= hi:
                result.update(range(lo, hi + 1))
        else:
            try:
                v = int(part)
            except ValueError:
                if warnings is not None:
                    warnings.append(f"layer route part {part!r} ignored")
                continue
            recognized = True
            if v < 0:
                v += num_blocks
            if 0 <= v < num_blocks:
                result.add(v)
    if result:
        return sorted(result)
    if recognized:
        if warnings is not None:
            warnings.append(
                f"layer route {str(text)!r} matches no blocks "
                f"(model has {num_blocks} layers)"
            )
        return []
    return None


def parse_timing_filter(text):
    """Parse ``start-end`` or ``start-end~fade`` into ``(start, end, fade)``.

    All values are sampling-progress percentages in [0, 1]. Bare values above
    1 are treated as percents (``0-45`` -> ``0.0-0.45``); the fade is scaled
    the same way. ``fade`` is the smoothstep ramp width applied on both sides
    of the window (0 = hard on/off). Returns None when unparseable.
    """
    if not text:
        return None
    s = str(text).strip().replace(" ", "")
    if not s:
        return None
    fade = 0.0
    if "~" in s:
        s, fade_text = s.split("~", 1)
        if "~" in fade_text:
            return None
        try:
            fade = float(fade_text)
        except ValueError:
            return None
        if fade != fade or fade < 0.0:  # NaN or negative
            return None
    if "-" not in s[1:]:
        return None
    dash_idx = s.index("-", 1)
    try:
        start = float(s[:dash_idx])
        end = float(s[dash_idx + 1:])
    except ValueError:
        return None
    if start != start or end != end:  # NaN
        return None
    if start > end:
        start, end = end, start
    if end > 1.0:
        # A window like 0-45 or 50-80 is a percent shorthand; rescale the whole
        # window (and fade) rather than clamping to the always-active [0, 1].
        start /= 100.0
        end /= 100.0
        fade /= 100.0
    if end > 1.0 or end <= start:
        return None
    start = clamp_float(start, 0.0, 1.0)
    end = clamp_float(end, 0.0, 1.0)
    fade = clamp_float(fade, 0.0, 0.5)
    return start, end, fade


def normalize_weights(weights):
    total = sum(abs(w) for w in weights)
    if total <= 1e-8:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def _format_prompt_weight(value):
    return f"{float(value):g}"


def build_direct_artist_prompt(names, weights, base_prompt):
    parts = []
    for idx, name in enumerate(names or []):
        text = str(name or "").strip()
        if not text:
            continue
        weight = float(weights[idx]) if idx < len(weights or []) else 1.0
        if weight < 0.0:
            raise ValueError(
                "prompt_passthrough does not support negative artist weight; "
                "negative weights are only available in the cross-attention mixer."
            )
        if abs(weight) <= 1e-8:
            continue
        if abs(weight - 1.0) <= 1e-8:
            parts.append(text)
        else:
            parts.append(f"({text}:{_format_prompt_weight(weight)})")
    base = str(base_prompt or "").strip()
    if base:
        parts.append(base)
    return ", ".join(parts)


def build_passthrough_prompt(raw_artist_chain, names, weights,
                             has_explicit_weights, base_prompt):
    raw = str(raw_artist_chain or "").strip()
    base = str(base_prompt or "").strip()
    if raw and not bool(has_explicit_weights):
        return f"{raw}\n\n{base}" if base else raw
    artists = build_direct_artist_prompt(names, weights, "")
    if artists and base:
        return f"{artists}\n\n{base}"
    if artists:
        return artists
    return base


def resolve_artist_layer_routes(route_texts, num_blocks, warnings=None):
    routes = []
    has_routes = False
    for route in route_texts or []:
        parsed = parse_layer_filter(route, num_blocks, warnings)
        if parsed is None:
            # No filter for this artist: active on every block.
            routes.append(None)
        else:
            # Either a concrete block set, or [] meaning "explicitly nowhere".
            has_routes = True
            routes.append(set(parsed))
    return routes, has_routes


def resolve_artist_timing_routes(timing_texts):
    timings = []
    has_timings = False
    for timing in timing_texts or []:
        parsed = parse_timing_filter(timing)
        if parsed is not None:
            has_timings = True
            timings.append(parsed)
        else:
            timings.append(None)
    return timings, has_timings


def resolve_target_blocks_from_options(adv, num_blocks, strict=False):
    layer_filter_text = str((adv or {}).get("layer_filter", "") or "")
    explicit_blocks = parse_layer_filter(layer_filter_text, num_blocks)
    if explicit_blocks is not None:
        # An explicit filter that resolves to no blocks means exactly that;
        # do not fall through to start/end (which would patch everything).
        if not explicit_blocks:
            if strict:
                raise ValueError(
                    f"[AnimaCrossAttn] layer_filter {layer_filter_text!r} matches "
                    f"no blocks (model has {num_blocks} blocks)"
                )
            return []
        return explicit_blocks
    sb = int((adv or {}).get("start_block", 0))
    eb = int((adv or {}).get("end_block", -1))
    sb_real = max(0, sb)
    eb_real = num_blocks - 1 if eb < 0 else min(num_blocks - 1, eb)
    if sb_real > eb_real:
        if strict:
            raise ValueError(
                f"[AnimaCrossAttn] start_block={sb_real} > end_block={eb_real} "
                f"(model has {num_blocks} blocks)"
            )
        return []
    return list(range(sb_real, eb_real + 1))


def expand_prompt_weights(text):
    """v26: Expand 'weight::target::' in base_prompt to ComfyUI standard bracket syntax '(target:weight)'.

    Syntax: Trailing :: explicitly marks target boundary (can span across commas).
      '1.5::masterpiece::, 1girl'                  → '(masterpiece:1.5), 1girl'
      '1.3::detailed background, intricate::, 1girl' → '(detailed background, intricate:1.3), 1girl'
      'masterpiece, 1.5::high quality::, ok'        → 'masterpiece, (high quality:1.5), ok'

    Non-matching cases (kept as-is for user visibility):
      '1.5::masterpiece'      (missing trailing ::)
      'abc::masterpiece::'    (prefix not a number)

    Only prefix syntax (weight::target::) is supported for base_prompt; postfix is not supported
    to avoid confusion with :: decoration text in sentences.
    Weight clamped to [0.0, 4.0] (same as strength range).
    Does not modify parenthesis syntax (name:1.5) — ComfyUI native syntax is passed through unchanged.
    """
    if not text or "::" not in text:
        return text

    result = []
    i = 0
    n = len(text)
    while i < n:
        sep = text.find("::", i)
        if sep < 0:
            result.append(text[i:])
            break

        # Look backward: find weight from sep leftward, boundary is comma/newline/text start
        # (full-width CJK comma ，included so weight prefixes survive after it).
        boundary_left = i
        for j in range(sep - 1, i - 1, -1):
            if text[j] in ",，\n\r":
                boundary_left = j + 1
                break

        weight_str = text[boundary_left:sep].strip()
        weight_val = None
        try:
            w = float(weight_str)
            weight_val = max(0.0, min(4.0, w))
        except ValueError:
            pass

        if weight_val is None:
            # Not a valid weight → skip this ::, keep original text
            result.append(text[i:sep + 2])
            i = sep + 2
            continue

        # Look forward: find trailing '::' after sep+2 (not allowed to cross newline)
        target_start = sep + 2
        while target_start < n and text[target_start] == " ":
            target_start += 1

        end_marker = -1
        k = target_start
        while k < n - 1:
            if text[k] == "\n" or text[k] == "\r":
                break
            if text[k] == ":" and text[k + 1] == ":":
                end_marker = k
                break
            k += 1

        if end_marker < 0:
            result.append(text[i:sep + 2])
            i = sep + 2
            continue

        target = text[target_start:end_marker].strip()
        if not target:
            result.append(text[i:sep + 2])
            i = sep + 2
            continue

        # Output: everything before boundary_left + expanded bracket syntax
        result.append(text[i:boundary_left])
        result.append(f"({target}:{weight_val:g})")
        i = end_marker + 2

    return "".join(result)
