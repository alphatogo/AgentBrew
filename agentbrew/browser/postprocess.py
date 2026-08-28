"""
Browser tool output post-processing.

Compresses playwright MCP tool outputs (ARIA accessibility trees) before
passing them to the LLM, reducing token consumption by 70-99% while
preserving all information needed for task completion.

Usage
-----
    from agentbrew.browser.postprocess import postprocess_tool_output

    raw = extract_text_from_result(tool_result)
    compressed = postprocess_tool_output(tool_name, raw)

Standalone test
---------------
    conda run -n mcp python -m agentbrew.browser.postprocess
"""

from __future__ import annotations

import re
from typing import List, Tuple


# ── Role classification ───────────────────────────────────────────────────────

# Always keep: carry real semantic value for text-only tasks.
_KEEP_ROLES = {
    # Navigation & interaction
    "link", "button", "input", "textbox", "searchbox", "search",
    "checkbox", "tab", "tabpanel", "menuitem",
    # Text content
    "heading", "paragraph", "article", "listitem", "list",
    "strong", "time", "code", "definition", "term",
    "blockquote", "note",
    # Tables — critical for sports stats, model leaderboards, ICLR papers
    "table", "rowgroup", "row", "cell", "columnheader", "rowheader", "gridcell",
    # Layout anchors
    "main", "banner", "navigation", "form", "region", "group",
    # Dialogs (cookie banners, modals) — LLM needs to dismiss them
    "dialog",
}

# Always skip regardless of text content.
_ALWAYS_SKIP_ROLES = {
    "option",        # dropdown items (20+ per combobox, pure noise)
    "radio",         # show/hide toggle labels
    "separator",     # visual dividers
    "scrollbar",     # chrome
    "contentinfo",   # footer (same boilerplate on every page)
    "img",           # tasks are text-only, no multimodal
    "figure",        # same
}

# Skip unless they carry direct inline text (>3 chars).
_SKIP_UNLESS_TEXT_ROLES = {"generic", "none", "presentation"}

# Combobox: keep the label but suppress option children.
_COMBOBOX_ROLES = {"combobox"}

# ── Attribute noise patterns ──────────────────────────────────────────────────

_ATTR_NOISE = re.compile(
    r'\s*\[cursor=pointer\]'   # implied for all links/buttons
    r'|\s*\[active\]'          # only one element is active per page
    r'|\s*\[level=\d+\]'       # heading level redundant
    r'|\s*\[selected\]'        # combobox selected state
)

# Large inline JSON blobs injected by SPAs (>80 chars in quoted text).
_JSON_BLOB = re.compile(r'\{["\w].*?\}|\[["\d{}\[\],:\s]{80,}\]', re.DOTALL)


def _clean_line(line: str) -> str:
    """Strip noisy bracketed attributes and large inline JSON from one line."""
    line = _ATTR_NOISE.sub("", line)

    def _strip_blob(m: re.Match) -> str:
        s = m.group(0)
        cleaned = _JSON_BLOB.sub("…", s)
        return cleaned if len(cleaned) < len(s) else s

    return re.sub(r'"[^"]{80,}"', _strip_blob, line)


# ── Core ARIA tree filter ─────────────────────────────────────────────────────

def _filter_aria_tree(yaml_block: str, max_depth: int = 8) -> str:
    """
    Filter a playwright ARIA accessibility tree (content inside ```yaml blocks).

    Rules applied in order per line:
    1. Hard depth gate (max_depth).
    2. Combobox collapse: keep label, suppress option children.
    3. /url: lines always kept (click targets).
    4. '- text: …' lines always kept (raw inline text nodes).
    5. _ALWAYS_SKIP_ROLES: dropped unconditionally.
    6. _KEEP_ROLES: kept unconditionally.
    7. _SKIP_UNLESS_TEXT_ROLES: kept only when carrying quoted text (>3 chars)
       or an inline colon-value.
    8. Unknown roles: kept if they have meaningful quoted text.
    """
    lines = yaml_block.splitlines()
    kept: List[str] = []
    inside_combobox_depth: int = -1

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        indent = len(line) - len(stripped)
        depth = indent // 2  # playwright uses 2-space indent

        if depth > max_depth:
            continue

        # Reset combobox suppression when depth returns to parent level
        if inside_combobox_depth >= 0 and depth <= inside_combobox_depth:
            inside_combobox_depth = -1

        # Drop option children inside combobox
        if inside_combobox_depth >= 0 and depth > inside_combobox_depth:
            continue

        # Clean noise attributes before any decision
        line = _clean_line(line)
        stripped = line.lstrip()

        # URL reference lines (needed for click targets)
        if stripped.startswith("/url:"):
            kept.append(line)
            continue

        # Raw text nodes
        if re.match(r"-\s+text:\s+", stripped) and re.search(r":\s+\S", stripped):
            kept.append(line)
            continue

        role_match = re.match(r"-\s+(\w[\w-]*)", stripped)
        role = role_match.group(1).lower() if role_match else ""

        if role in _ALWAYS_SKIP_ROLES:
            continue

        if role in _COMBOBOX_ROLES:
            kept.append(line)
            inside_combobox_depth = depth
            continue

        if role in _KEEP_ROLES:
            kept.append(line)
            continue

        if role in _SKIP_UNLESS_TEXT_ROLES:
            has_quoted = re.search(r'"[^"]{3,}"', stripped)
            has_inline = re.search(r'\]:\s+\S{2,}', stripped)
            if has_quoted or has_inline:
                kept.append(line)
            continue

        # Unknown role: keep if meaningful quoted text
        if re.search(r'"[^"]{3,}"', stripped):
            kept.append(line)

    # Collapse consecutive blank lines
    result: List[str] = []
    for line in kept:
        if line == "" and result and result[-1] == "":
            continue
        result.append(line)

    return "\n".join(result).strip()


# ── Output assembly helpers ───────────────────────────────────────────────────

def _extract_page_header(raw: str) -> str:
    """Extract the ### Page section (URL, Title, Console lines)."""
    lines, in_page = [], False
    for line in raw.splitlines():
        if line.strip() == "### Page":
            in_page = True
            lines.append(line)
            continue
        if in_page:
            if line.startswith("###"):
                break
            lines.append(line)
    return "\n".join(lines).strip()


def _extract_snapshot_yaml(raw: str) -> Tuple[str, str]:
    """
    Split raw playwright output into (header, yaml_block).
    yaml_block is the content inside ```yaml … ```.
    """
    m = re.search(r"```yaml\s*\n(.*?)```", raw, re.DOTALL)
    if not m:
        return raw, ""
    return raw[: m.start()].rstrip(), m.group(1)


def _assemble(page_header: str, filtered_yaml: str, max_chars: int) -> str:
    """Combine page header + filtered ARIA tree, then hard-truncate."""
    parts: List[str] = []
    if page_header:
        parts.append(page_header)
        parts.append("")
    parts += ["```yaml", filtered_yaml, "```"]
    result = "\n".join(parts)
    if len(result) > max_chars:
        hidden = len(result) - max_chars
        result = result[:max_chars] + f"\n\n... [POSTPROCESS: {hidden} chars truncated]"
    return result


# ── Per-tool post-processors ──────────────────────────────────────────────────

def postprocess_snapshot(raw: str, max_chars: int = 8000) -> str:
    """
    Post-process browser_snapshot / browser_navigate output.

    Always applies ARIA filtering (removes noisy attributes and irrelevant roles)
    regardless of content length, then hard-truncates to max_chars.

    max_chars notes:
    - 8000 (default): covers most single-purpose pages.
    - Long listing pages (EMNLP 764 papers, NeurIPS 2000+ papers): first award
      entry appears at ~17 KB into the filtered tree. For these the LLM should
      use browser_evaluate() / browser_run_code() to search rather than relying
      on the full snapshot.
    - JS-loaded content (Premier League stats, CVPR pricing): never appear in
      the ARIA tree regardless of max_chars. Use browser_evaluate().
    """
    if not raw:
        return raw
    page_header = _extract_page_header(raw)
    _, yaml_block = _extract_snapshot_yaml(raw)
    if not yaml_block:
        if len(raw) > max_chars:
            return raw[:max_chars] + f"\n\n... [POSTPROCESS: {len(raw) - max_chars} chars truncated]"
        return raw
    filtered = _filter_aria_tree(yaml_block)
    return _assemble(page_header, filtered, max_chars)


def postprocess_navigate(raw: str, max_chars: int = 8000) -> str:
    """browser_navigate returns the same format as browser_snapshot."""
    return postprocess_snapshot(raw, max_chars)


def postprocess_click(raw: str, max_chars: int = 6000) -> str:
    """
    browser_click returns: ### Ran Playwright code + ### Page + ### Snapshot yaml.
    Keep the JS confirmation lines, the page header, and the filtered ARIA tree.
    """
    page_header = _extract_page_header(raw)
    _, yaml_block = _extract_snapshot_yaml(raw)

    # Collect only the JS action confirmation (stop before first ### Page or ### Snapshot)
    conf_lines = []
    for line in raw.splitlines():
        if line.startswith("### Page") or line.startswith("### Snapshot") or line.startswith("```yaml"):
            break
        conf_lines.append(line)
    confirmation = "\n".join(conf_lines).strip()

    parts = []
    if confirmation:
        parts.append(confirmation)
    if page_header:
        parts.append(page_header)

    if yaml_block:
        filtered = _filter_aria_tree(yaml_block)
        parts += ["", "```yaml", filtered, "```"]

    result = "\n".join(parts)
    if len(result) > max_chars:
        hidden = len(result) - max_chars
        result = result[:max_chars] + f"\n\n... [POSTPROCESS: {hidden} chars truncated]"
    return result


def postprocess_type(raw: str, max_chars: int = 6000) -> str:
    """browser_type / browser_fill_form / browser_select_option."""
    return postprocess_click(raw, max_chars)


def postprocess_evaluate(raw: str, max_chars: int = 4000) -> str:
    """browser_evaluate / browser_run_code returns a JS result."""
    if len(raw) > max_chars:
        return raw[:max_chars] + f"\n... [POSTPROCESS: {len(raw) - max_chars} chars truncated]"
    return raw.strip()


def postprocess_run_code(raw: str, max_chars: int = 6000) -> str:
    """
    browser_run_code returns: ### Result\\n<JS output>\\n### Ran Playwright code\\n<JS code>\\n### Page...

    Priority: keep ### Result fully, then ### Page (URL/title), then truncate
    the verbose JS code section to save tokens.
    """
    if not raw:
        return raw

    lines = raw.splitlines()
    sections: List[Tuple[str, List[str]]] = []  # (header, lines)
    current_header = ""
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("### "):
            if current_lines or current_header:
                sections.append((current_header, current_lines))
            current_header = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_header or current_lines:
        sections.append((current_header, current_lines))

    # Priority: ### Result (full) > ### Page (full) > others (truncated)
    # ### Ran Playwright code — verbose JS, truncate aggressively
    # ### Events — console log noise, keep only a short summary
    KEEP_FULL = {"### Result", "### Page"}
    CODE_MAX = 300    # keep first N chars of the JS code block
    EVENTS_MAX = 400  # keep first N chars of events (enough to see main events)

    parts: List[str] = []
    for header, sec_lines in sections:
        sec_text = "\n".join(sec_lines).strip()
        if header in KEEP_FULL or not header:
            if header:
                parts.append(header)
            if sec_text:
                parts.append(sec_text)
        elif header == "### Events":
            if header:
                parts.append(header)
            if sec_text:
                if len(sec_text) > EVENTS_MAX:
                    parts.append(sec_text[:EVENTS_MAX] + "\n... [POSTPROCESS: events truncated]")
                else:
                    parts.append(sec_text)
        else:
            # Truncate verbose sections (Ran Playwright code, etc.)
            if header:
                parts.append(header)
            if sec_text:
                if len(sec_text) > CODE_MAX:
                    parts.append(sec_text[:CODE_MAX] + "\n... [POSTPROCESS: code truncated]")
                else:
                    parts.append(sec_text)

    result = "\n".join(parts)
    if len(result) > max_chars:
        hidden = len(result) - max_chars
        result = result[:max_chars] + f"\n\n... [POSTPROCESS: {hidden} chars truncated]"
    return result.strip()


def postprocess_network_requests(raw: str, max_chars: int = 3000) -> str:
    """
    browser_network_requests returns a log of all HTTP requests (often 50-100+ KB).

    Filter to show only requests to the primary domain (not CDN/fonts/analytics)
    and only the first max_lines entries, then hard-truncate.
    """
    if not raw or len(raw) <= max_chars:
        return raw.strip() if raw else raw

    lines = raw.splitlines()
    kept: List[str] = []
    CDN_PATTERNS = re.compile(
        r'(cdn\.|fonts\.|googleapis\.|gstatic\.|cloudflare\.|jquery\.|d3js\.|mathjax\.|fontawesome)',
        re.IGNORECASE
    )

    for line in lines:
        if line.startswith("### "):
            kept.append(line)
            continue
        # Keep only non-CDN requests
        if CDN_PATTERNS.search(line):
            continue
        kept.append(line)

    result = "\n".join(kept)
    if len(result) > max_chars:
        hidden = len(result) - max_chars
        result = result[:max_chars] + f"\n\n... [POSTPROCESS: {hidden} chars truncated]"
    return result.strip()


def postprocess_noop(raw: str) -> str:
    """Pass-through for tools whose output is already short."""
    return raw.strip()


_DISPATCH: dict = {
    "browser_navigate":           postprocess_navigate,
    "browser_snapshot":           postprocess_snapshot,
    "browser_click":              postprocess_click,
    "browser_type":               postprocess_type,
    "browser_fill_form":          postprocess_type,
    "browser_select_option":      postprocess_type,
    "browser_press_key":          postprocess_click,   # press key may trigger page change
    "browser_evaluate":           postprocess_evaluate,
    "browser_run_code":           postprocess_run_code,
    "browser_network_requests":   postprocess_network_requests,
    "browser_wait_for":           postprocess_noop,
    "browser_close":              postprocess_noop,
    "browser_navigate_back":      postprocess_navigate,
    "browser_hover":              postprocess_click,
    "browser_drag":               postprocess_click,
}


def postprocess_tool_output_gpt(tool_name: str, raw: str) -> str:
    """
    GPT-validated main entry point (original version, do NOT modify).

    Applies the appropriate post-processor for the given playwright tool name.
    Falls back to identity for unknown tools (non-browser tools are returned
    unchanged so this function is safe to call on any tool result).

    Parameters
    ----------
    tool_name : str
        The playwright MCP tool name, e.g. "browser_navigate".
    raw : str
        The raw text extracted from the MCP CallToolResult.

    Returns
    -------
    str
        Compressed output safe to pass directly to the LLM.
    """
    fn = _DISPATCH.get(tool_name)
    if fn is None:
        return raw  # not a browser tool — return unchanged
    return fn(raw)


# ── 32B-adapted ARIA filter ───────────────────────────────────────────────────

def _filter_aria_tree_32b(yaml_block: str, max_depth: int = 12, max_listitems: int = 60) -> str:
    """
    32B-adapted ARIA tree filter with deeper traversal and relaxed generic filtering.

    Changes vs GPT version:
    - max_depth default raised to 12 (was 8): exposes paper/article content
      nested deeply in SPA pages (OpenReview, HuggingFace).
    - generic/none/presentation: kept whenever they carry ANY colon-value inline
      text, even without quoted text — catches nav-bar labels and space titles
      that appear as plain `generic: Label` lines.
    - max_listitems=60: caps total listitem count to prevent explosion on "all
      papers" listing pages (e.g. iclr.cc/virtual/2025/papers.html has 1000+
      papers × deep author links = 320k chars without this cap).
      60 covers: nav menus (~15) + search results (~7) + per-page paper list
      (~25) with room to spare, while cutting runaway listing pages.
    """
    lines = yaml_block.splitlines()
    kept: List[str] = []
    inside_combobox_depth: int = -1
    # Listitem explosion guard
    listitem_count: int = 0
    suppressed_listitem_depth: int = -1

    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue

        indent = len(line) - len(stripped)
        depth = indent // 2

        if depth > max_depth:
            continue

        # Reset combobox suppression
        if inside_combobox_depth >= 0 and depth <= inside_combobox_depth:
            inside_combobox_depth = -1
        if inside_combobox_depth >= 0 and depth > inside_combobox_depth:
            continue

        # Reset suppressed-listitem scope when we leave it
        if suppressed_listitem_depth >= 0 and depth <= suppressed_listitem_depth:
            suppressed_listitem_depth = -1
        # Skip children of a suppressed listitem
        if suppressed_listitem_depth >= 0 and depth > suppressed_listitem_depth:
            continue

        line = _clean_line(line)
        stripped = line.lstrip()

        if stripped.startswith("/url:"):
            kept.append(line)
            continue

        if re.match(r"-\s+text:\s+", stripped) and re.search(r":\s+\S", stripped):
            kept.append(line)
            continue

        role_match = re.match(r"-\s+(\w[\w-]*)", stripped)
        role = role_match.group(1).lower() if role_match else ""

        if role in _ALWAYS_SKIP_ROLES:
            continue

        # Listitem cap: count every listitem; suppress children once over budget
        if role == "listitem":
            listitem_count += 1
            if listitem_count > max_listitems:
                if listitem_count == max_listitems + 1:
                    kept.append(
                        f"... [ARIA: list continues — {listitem_count - 1}+ items shown, "
                        f"remaining items suppressed to stay within context budget]"
                    )
                suppressed_listitem_depth = depth
                continue  # don't include this listitem or its children
            # Within budget: fall through to normal _KEEP_ROLES handling below

        if role in _COMBOBOX_ROLES:
            kept.append(line)
            inside_combobox_depth = depth
            continue

        if role in _KEEP_ROLES:
            kept.append(line)
            continue

        if role in _SKIP_UNLESS_TEXT_ROLES:
            has_quoted = re.search(r'"[^"]{3,}"', stripped)
            # 32B relaxation: also keep if there is any colon-separated inline value
            has_inline = re.search(r'\]:\s+\S{2,}', stripped)
            # 32B relaxation: also keep plain "role: SomeText" lines (nav labels, etc.)
            has_plain_value = re.search(r':\s+\S{2,}', stripped)
            if has_quoted or has_inline or has_plain_value:
                kept.append(line)
            continue

        if re.search(r'"[^"]{3,}"', stripped):
            kept.append(line)

    result: List[str] = []
    for line in kept:
        if line == "" and result and result[-1] == "":
            continue
        result.append(line)

    return "\n".join(result).strip()


# ── 32B per-tool post-processors ──────────────────────────────────────────────

def postprocess_snapshot_32b(raw: str, max_chars: int = 14000) -> str:
    """
    32B-adapted snapshot: larger char budget (14 000) + deeper ARIA traversal.

    Rationale: HuggingFace pages are ~12 000 chars after GPT-style filtering
    (4 000 chars were being truncated), hiding the top-nav Blog link and other
    navigation landmarks. OpenReview paper lists nest titles at depth 9+.
    """
    if not raw:
        return raw
    page_header = _extract_page_header(raw)
    _, yaml_block = _extract_snapshot_yaml(raw)
    if not yaml_block:
        if len(raw) > max_chars:
            return raw[:max_chars] + f"\n\n... [POSTPROCESS: {len(raw) - max_chars} chars truncated]"
        return raw
    filtered = _filter_aria_tree_32b(yaml_block)
    return _assemble(page_header, filtered, max_chars)


def postprocess_navigate_32b(raw: str, max_chars: int = 20000) -> str:
    """browser_navigate returns the same format as browser_snapshot.

    Larger budget than snapshot (20 000) to handle API JSON pages (e.g.
    huggingface.co/api/models/<model>) where safetensors.total appears
    near the end of a ~17 KB response.
    """
    return postprocess_snapshot_32b(raw, max_chars)


def postprocess_click_32b(raw: str, max_chars: int = 10000) -> str:
    """
    32B-adapted click: larger char budget (10 000) + deeper ARIA traversal.

    After a click the resulting page snapshot needs the same headroom as a
    fresh navigate so the model can see what changed.
    """
    page_header = _extract_page_header(raw)
    _, yaml_block = _extract_snapshot_yaml(raw)

    conf_lines = []
    for line in raw.splitlines():
        if line.startswith("### Page") or line.startswith("### Snapshot") or line.startswith("```yaml"):
            break
        conf_lines.append(line)
    confirmation = "\n".join(conf_lines).strip()

    parts = []
    if confirmation:
        parts.append(confirmation)
    if page_header:
        parts.append(page_header)

    if yaml_block:
        filtered = _filter_aria_tree_32b(yaml_block)
        parts += ["", "```yaml", filtered, "```"]

    result = "\n".join(parts)
    if len(result) > max_chars:
        hidden = len(result) - max_chars
        result = result[:max_chars] + f"\n\n... [POSTPROCESS: {hidden} chars truncated]"
    return result


def postprocess_type_32b(raw: str, max_chars: int = 10000) -> str:
    """browser_type / browser_fill_form / browser_select_option."""
    return postprocess_click_32b(raw, max_chars)


def postprocess_evaluate_32b(raw: str, max_chars: int = 6000) -> str:
    """browser_evaluate / browser_run_code — larger budget for JS results."""
    if len(raw) > max_chars:
        return raw[:max_chars] + f"\n... [POSTPROCESS: {len(raw) - max_chars} chars truncated]"
    return raw.strip()


def postprocess_run_code_32b(raw: str, max_chars: int = 10000) -> str:
    """
    32B-adapted run_code: larger overall budget (10 000).

    Keeps the same section priorities as the GPT version (Result > Page > code).
    """
    if not raw:
        return raw

    lines = raw.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_header = ""
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("### "):
            if current_lines or current_header:
                sections.append((current_header, current_lines))
            current_header = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_header or current_lines:
        sections.append((current_header, current_lines))

    KEEP_FULL = {"### Result", "### Page"}
    CODE_MAX = 500    # slightly larger than GPT version (300)
    EVENTS_MAX = 600

    parts: List[str] = []
    for header, sec_lines in sections:
        sec_text = "\n".join(sec_lines).strip()
        if header in KEEP_FULL or not header:
            if header:
                parts.append(header)
            if sec_text:
                parts.append(sec_text)
        elif header == "### Events":
            if header:
                parts.append(header)
            if sec_text:
                if len(sec_text) > EVENTS_MAX:
                    parts.append(sec_text[:EVENTS_MAX] + "\n... [POSTPROCESS: events truncated]")
                else:
                    parts.append(sec_text)
        else:
            if header:
                parts.append(header)
            if sec_text:
                if len(sec_text) > CODE_MAX:
                    parts.append(sec_text[:CODE_MAX] + "\n... [POSTPROCESS: code truncated]")
                else:
                    parts.append(sec_text)

    result = "\n".join(parts)
    if len(result) > max_chars:
        hidden = len(result) - max_chars
        result = result[:max_chars] + f"\n\n... [POSTPROCESS: {hidden} chars truncated]"
    return result.strip()


_DISPATCH_32B: dict = {
    "browser_navigate":           postprocess_navigate_32b,
    "browser_snapshot":           postprocess_snapshot_32b,
    "browser_click":              postprocess_click_32b,
    "browser_type":               postprocess_type_32b,
    "browser_fill_form":          postprocess_type_32b,
    "browser_select_option":      postprocess_type_32b,
    "browser_press_key":          postprocess_click_32b,
    "browser_evaluate":           postprocess_evaluate_32b,
    "browser_run_code":           postprocess_run_code_32b,
    "browser_network_requests":   postprocess_network_requests,   # unchanged
    "browser_wait_for":           postprocess_noop,
    "browser_close":              postprocess_noop,
    "browser_navigate_back":      postprocess_navigate_32b,
    "browser_hover":              postprocess_click_32b,
    "browser_drag":               postprocess_click_32b,
}


def postprocess_tool_output(tool_name: str, raw: str) -> str:
    """
    Main entry point — 32B-adapted version.

    Uses _DISPATCH_32B which applies deeper ARIA traversal (max_depth=12)
    and larger char budgets (snapshot/navigate: 14 000, click/type: 10 000,
    evaluate: 6 000, run_code: 10 000) compared to the GPT version.

    The GPT-validated original is preserved as postprocess_tool_output_gpt().

    Parameters
    ----------
    tool_name : str
        The playwright MCP tool name, e.g. "browser_navigate".
    raw : str
        The raw text extracted from the MCP CallToolResult.

    Returns
    -------
    str
        Compressed output safe to pass directly to the LLM.
    """
    fn = _DISPATCH_32B.get(tool_name)
    if fn is None:
        return raw  # not a browser tool — return unchanged
    return fn(raw)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import json
    import shutil
    from contextlib import AsyncExitStack

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("mcp package not found. Run: conda run -n mcp python -m agentbrew.browser.postprocess")
        raise

    def _extract_text(result) -> str:
        if hasattr(result, "content"):
            parts = []
            for item in result.content:
                if hasattr(item, "text") and item.text:
                    parts.append(item.text)
                elif hasattr(item, "resource") and hasattr(item.resource, "text"):
                    parts.append(item.resource.text)
            return "\n".join(parts)
        return str(result)

    DEMO_CALLS = [
        ("browser_navigate", {"url": "https://arxiv.org/abs/2502.13923"}),
        ("browser_navigate", {"url": "https://arxiv.org/search/?searchtype=all&query=Qwen2.5-VL"}),
        ("browser_navigate", {"url": "https://huggingface.co/Rhymes-AI/Aria"}),
        ("browser_navigate", {"url": "https://news.ycombinator.com/"}),
        ("browser_close",    {}),
    ]

    async def _run():
        command = shutil.which("npx") or "npx"
        params = StdioServerParameters(
            command=command,
            args=["@playwright/mcp@latest", "--headless", "--isolated",
                  "--browser", "chromium", "--image-responses", "omit"],
        )
        print("=" * 70)
        print("agentbrew.browser.postprocess — standalone test")
        print("=" * 70)
        results = []
        async with AsyncExitStack() as stack:
            stdio = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(*stdio))
            await session.initialize()
            for tool, args in DEMO_CALLS:
                print(f"\n>>> {tool}({json.dumps(args)[:60]})")
                try:
                    r = await session.call_tool(tool, args)
                    raw = _extract_text(r)
                    proc = postprocess_tool_output(tool, raw)
                    ratio = len(proc) / max(len(raw), 1) * 100
                    print(f"  raw={len(raw):,}  post={len(proc):,}  saved={100-ratio:.0f}%")
                    results.append((tool, len(raw), len(proc)))
                except Exception as e:
                    print(f"  ERROR: {e}")

        total_raw = sum(r for _, r, _ in results)
        total_post = sum(p for _, _, p in results)
        print(f"\nTotal: {total_raw:,} → {total_post:,} chars  "
              f"({(1-total_post/max(total_raw,1))*100:.1f}% saved)")

    asyncio.run(_run())
