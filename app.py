import io
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import streamlit as st
from pypdf import PdfReader, PdfWriter

# ---------- Helpers ----------
@dataclass
class LabelInfo:
    box_index: int
    total_boxes: Optional[int]
    shipment_id: Optional[str]
    fnsku: Optional[str]
    asin: Optional[str]
    sku: Optional[str]
    tracking: Optional[str]

FNSKU_RE = re.compile(r"\bX[A-Z0-9]{10}\b")
ASIN_RE = re.compile(r"\bB[0-9A-Z]{9}\b")
SHIPMENT_RE = re.compile(r"\bFBA[ A-]*[A-Z0-9]{6,}\b")
CARTON_RE = re.compile(r"(\d+)\s*/?\s*of\s*/?\s*(\d+)", re.IGNORECASE)
SKU_LINE_RE = re.compile(r"(?:(?:MSKU|SKU)\s*[:#-]?\s*)([A-Za-z0-9._\-]+)")
# Common tracking formats (DHL, UPS, FedEx)
TRACKING_RES = [
    re.compile(r"\bJJD[0-9A-Z]{10,}\b", re.IGNORECASE),   # DHL JJD...
    re.compile(r"\b\d{10,12}\b"),                         # 10-12 digits (DHL/others)
    re.compile(r"\b1Z[0-9A-Z]{16}\b", re.IGNORECASE),     # UPS 1Z...
    re.compile(r"\b\d{12,15}\b"),                        # fallback digits
]


def split_pdf_odd_even(src_bytes: bytes) -> Tuple[bytes, bytes, List[str]]:
    reader = PdfReader(io.BytesIO(src_bytes))
    odd_writer, even_writer = PdfWriter(), PdfWriter()
    texts: List[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        texts.append(text)
        # 0-based i: odd pages are 1,3,5... -> indices 0,2,4...
        if (i % 2) == 0:
            odd_writer.add_page(page)
        else:
            even_writer.add_page(page)

    odd_buf, even_buf = io.BytesIO(), io.BytesIO()
    odd_writer.write(odd_buf)
    even_writer.write(even_buf)
    return odd_buf.getvalue(), even_buf.getvalue(), texts


def extract_tracking(text: str) -> Optional[str]:
    for rx in TRACKING_RES:
        m = rx.search(text)
        if m:
            return m.group(0)
    return None


def extract_fba_fields(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Tuple[int, Optional[int]]]]:
    fnsku = (FNSKU_RE.search(text) or (None,))[0] if FNSKU_RE.search(text) else None
    asin = (ASIN_RE.search(text) or (None,))[0] if ASIN_RE.search(text) else None
    shipment_id = (SHIPMENT_RE.search(text) or (None,))[0] if SHIPMENT_RE.search(text) else None

    carton = None
    m = CARTON_RE.search(text)
    if m:
        try:
            idx = int(m.group(1))
            total = int(m.group(2))
            carton = (idx, total)
        except Exception:
            carton = None

    # SKU can be varied; try explicit label first
    sku = None
    m2 = SKU_LINE_RE.search(text)
    if m2:
        sku = m2.group(1)
    else:
        # Weak fallback: look for a likely SKU-like token near FNSKU/ASIN lines
        pass

    return fnsku, asin, sku, carton


def build_summary(fba_texts: List[str], ship_texts: List[str]) -> List[LabelInfo]:
    n = min(len(fba_texts), len(ship_texts))
    summary: List[LabelInfo] = []

    total_boxes_hint: Optional[int] = None
    for i in range(n):
        f_text = fba_texts[i]
        s_text = ship_texts[i]

        fnsku, asin, sku, carton = extract_fba_fields(f_text)
        tracking = extract_tracking(s_text)
        shipment_id = (SHIPMENT_RE.search(f_text) or SHIPMENT_RE.search(s_text))
        shipment_id_val = shipment_id.group(0) if shipment_id else None

        box_idx = i + 1
        total_boxes = None
        if carton:
            box_idx = carton[0]
            total_boxes = carton[1]
            total_boxes_hint = total_boxes
        else:
            total_boxes = total_boxes_hint

        summary.append(LabelInfo(
            box_index=box_idx,
            total_boxes=total_boxes,
            shipment_id=shipment_id_val,
            fnsku=fnsku,
            asin=asin,
            sku=sku,
            tracking=tracking,
        ))
    return summary


def to_csv_bytes(rows: List[LabelInfo]) -> bytes:
    import csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(asdict(rows[0]).keys()) if rows else [
        "box_index","total_boxes","shipment_id","fnsku","asin","sku","tracking"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow(asdict(r))
    return buf.getvalue().encode("utf-8")


# ---------- UI ----------
st.set_page_config(page_title="FBA & Shipping Labels Splitter", page_icon="📦", layout="centered")
st.title("📦 FBA & Shipping Labels Splitter")
st.write("Upload the single alternating-labels PDF from Amazon (FBA label, then shipping label, etc.). We'll split it into two PDFs and generate a CSV summary.")

with st.expander("Options", expanded=False):
    assume_fba_is_odd = st.checkbox("Assume FBA labels are on odd pages (1,3,5,...) (FBA, then Shipping)", value=True)
    st.caption("Disable this if your input starts with a shipping label instead.")
    custom_title = st.text_input("Summary title", value="Amazon order")
    show_ids = st.multiselect("Show IDs in lines", options=["FNSKU","ASIN","Tracking","Shipment ID"], default=["ASIN"])  # which identifiers to include per line

uploaded = st.file_uploader("Upload combined PDF", type=["pdf"])

if uploaded is not None:
    data = uploaded.read()
    odd_bytes, even_bytes, texts = split_pdf_odd_even(data)

    # Decide which is FBA/shipping depending on the first page assumption
    if assume_fba_is_odd:
        fba_pdf, ship_pdf = odd_bytes, even_bytes
        fba_texts = [t for i, t in enumerate(texts) if i % 2 == 0]
        ship_texts = [t for i, t in enumerate(texts) if i % 2 == 1]
    else:
        fba_pdf, ship_pdf = even_bytes, odd_bytes
        fba_texts = [t for i, t in enumerate(texts) if i % 2 == 1]
        ship_texts = [t for i, t in enumerate(texts) if i % 2 == 0]

    # Build summary
    summary_rows = build_summary(fba_texts, ship_texts)

    # ----- Build human-readable text summary -----
    def choose_id(row):
        # pick preferred identifier(s) to show
        parts = []
        if "FNSKU" in show_ids and row.fnsku:
            parts.append(row.fnsku)
        if "ASIN" in show_ids and row.asin:
            parts.append(row.asin)
        if "Tracking" in show_ids and row.tracking:
            parts.append(row.tracking)
        if "Shipment ID" in show_ids and row.shipment_id:
            parts.append(row.shipment_id)
        return " / ".join(parts)

    # choose the key used for grouping consecutive boxes (SKU preferred, else FNSKU, else ASIN)
    def label_key(row):
        return row.sku or row.fnsku or row.asin or "(Unknown)"

    # Group consecutive boxes with the same key
    groups = []  # list of (start_idx, end_idx, key, id_text)
    if summary_rows:
        start = 1
        prev_key = label_key(summary_rows[0])
        prev_id = choose_id(summary_rows[0])
        for i in range(1, len(summary_rows)):
            k = label_key(summary_rows[i])
            idt = choose_id(summary_rows[i])
            if k != prev_key or idt != prev_id:
                groups.append((start, i, prev_key, prev_id))
                start = i + 1
                prev_key, prev_id = k, idt
        groups.append((start, len(summary_rows), prev_key, prev_id))

    total_boxes = summary_rows[-1].total_boxes if summary_rows and summary_rows[-1].total_boxes else len(summary_rows)

    lines = [f"{custom_title}. Total of {total_boxes} Boxes.", "", "Order:"]
    for (a,b,key,ids) in groups:
        if a==b:
            rng = f"Box {a}:"
        else:
            rng = f"Box {a}-{b}:"
        suffix = f" / {ids}" if ids else ""
        # try to extract multiplier like '15x' from SKU if present at start; otherwise omit
        lines.append(f"{rng} {key}{suffix}")
    text_summary = "
".join(lines)

    st.subheader("Downloads")
    st.download_button("⬇️ Download FBA labels PDF", data=fba_pdf, file_name="fba_labels.pdf", mime="application/pdf")
    st.download_button("⬇️ Download Shipping labels PDF", data=ship_pdf, file_name="shipping_labels.pdf", mime="application/pdf")

    if summary_rows:
        # Offer CSV (still available) and Text summary
        csv_bytes = to_csv_bytes(summary_rows)
        st.download_button("⬇️ Download summary CSV", data=csv_bytes, file_name="labels_summary.csv", mime="text/csv")

        st.subheader("Text summary")
        st.code(text_summary)
        st.download_button("⬇️ Download summary TXT", data=text_summary.encode("utf-8"), file_name="labels_summary.txt", mime="text/plain")

        st.subheader("Raw data preview")
        st.dataframe([asdict(r) for r in summary_rows])

    st.info("If some fields come out empty, it's usually because the label is rendered as an image and doesn't contain extractable text. You can still use the odd/even PDFs. For OCR add-ons, see the notes in the sidebar.")

with st.sidebar:
    st.header("Notes & Tips")
    st.markdown(
        """
- **Pairs assumed**: Each FBA label corresponds to the immediately following shipping label.
- **Text extraction**: This app reads embedded text. Some PDFs are image-only; for those, consider running OCR first (e.g., **Adobe**, **Tesseract**, or your scanner settings).
- **What we try to read**: FNSKU (starts with **X**), ASIN (**B**********), optional **SKU/MSKU** lines, **carton number** (like `3 of 10`), and common **tracking** formats (DHL/UPS).
- **First page toggle**: If your file starts with a shipping label, uncheck the odd-page FBA assumption.
- **Privacy**: Everything runs locally in your browser session when served from your machine/server.
        """
    )
