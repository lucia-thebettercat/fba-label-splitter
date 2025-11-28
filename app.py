import io
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict
import streamlit as st
from pypdf import PdfReader, PdfWriter
import pandas as pd

# ---------- Google Sheets Integration ----------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_packaging_data_from_sheets() -> Dict[str, dict]:
    """
    Load packaging data from Google Sheets.
    Returns a dictionary mapping SKU to packaging info.
    """
    try:
        # Google Sheet URL (publicly accessible)
        sheet_url = "https://docs.google.com/spreadsheets/d/1ZsJJOt7P9bWubJk8K_ZLCZWR5Sw989P5wX4Bpu1ctDk/export?format=csv&gid=0"
        
        # Read the sheet
        df = pd.read_csv(sheet_url)
        
        # Create dictionary mapping SKU to packaging info
        packaging_dict = {}
        for _, row in df.iterrows():
            sku = row.get('TBC SKU reference', '').strip()
            if not sku or pd.isna(sku):
                continue
                
            # Column H: Weight per Full
            weight = row.get('Weight per Full', None)
            # Column L: Is the Product Fully Enclosed in a Closed Carton? (Yes/No)
            # Try both possible column names
            fully_enclosed = row.get('Is the Product Fully Enclosed in a Closed Carton? (Yes/No)', '')
            if pd.isna(fully_enclosed) or not fully_enclosed:
                # Try index-based access for column L (index 11, since columns are 0-indexed)
                try:
                    fully_enclosed = str(row.iloc[11]).strip() if len(row) > 11 else ''
                except:
                    fully_enclosed = ''
            else:
                fully_enclosed = str(fully_enclosed).strip()
            
            notes = row.get('Notes (e.g., open tray, partial lid, etc.)', '')
            
            # Handle weight (could be numeric or string)
            try:
                if pd.notna(weight):
                    # Remove any text like 'kg' or 'g' and convert to float
                    weight_str = str(weight).replace('kg', '').replace('g', '').strip()
                    weight_val = float(weight_str)
                else:
                    weight_val = None
            except:
                weight_val = None
            
            packaging_dict[sku] = {
                'weight': weight_val,
                'fully_enclosed': fully_enclosed,
                'notes': notes if pd.notna(notes) else ''
            }
        
        return packaging_dict
    except Exception as e:
        st.warning(f"⚠️ Could not load packaging data from Google Sheets: {e}")
        return {}


def determine_packaging_instruction(sku: str, packaging_data: Dict[str, dict]) -> Tuple[str, str]:
    """
    Determine whether to use master carton or repack into HIVE boxes.
    Returns a tuple: (instruction_type, instruction_text)
    
    instruction_type: "master", "repack", or "unknown"
    """
    if not packaging_data:
        return "unknown", "please verify packaging requirements"
    
    # First try exact match
    if sku in packaging_data:
        data = packaging_data[sku]
    else:
        # Try fuzzy matching - remove suffixes like -1, -2, etc.
        base_sku = sku.rsplit('-', 1)[0] if '-' in sku else sku
        
        # Look for matches
        matched_data = None
        for sheet_sku, sheet_data in packaging_data.items():
            # Check if base SKU matches or if sheet SKU is contained in the PDF SKU
            if base_sku == sheet_sku or sheet_sku in sku or sku.startswith(sheet_sku):
                matched_data = sheet_data
                break
            # Special case for "TBC_Other_Stanley" - match any variant
            if "TBC_Other_Stanley" in sheet_sku and "TBC_Other_Stanley" in sku:
                matched_data = sheet_data
                break
        
        if not matched_data:
            return "unknown", "please verify packaging requirements"
        
        data = matched_data
    weight = data.get('weight')
    fully_enclosed = str(data.get('fully_enclosed', '')).strip().lower()
    notes = data.get('notes', '')
    
    # Special case: products not sent by pallet
    if notes and 'not sent by pallet' in notes.lower():
        return "master", "product is not sent by pallet, only by carton"
    
    # Handle missing weight data
    if weight is None or pd.isna(weight):
        return "unknown", "weight data not available - please verify with supplier"
    
    # Amazon requirements: must be under 10kg and fully enclosed
    # Weight is assumed to be in grams if > 100, otherwise in kg
    weight_kg = weight if weight <= 100 else weight / 1000
    
    is_fully_enclosed = fully_enclosed in ['yes', 'y', 'true', '1']
    
    if weight_kg <= 10 and is_fully_enclosed:
        return "master", "Please ship with master carton, don't repack"
    else:
        return "repack", "Please repack on HIVE carton, ensure that weight is under 10kg"


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
TRACKING_RES = [
    re.compile(r"\bJJD[0-9A-Z]{10,}\b", re.IGNORECASE),  # DHL JJD...
    re.compile(r"\b\d{10,12}\b"),                        # 10-12 digits
    re.compile(r"\b1Z[0-9A-Z]{16}\b", re.IGNORECASE),    # UPS 1Z...
    re.compile(r"\b\d{12,15}\b"),                        # fallback digits
]


def split_pdf_odd_even(src_bytes: bytes) -> Tuple[bytes, bytes, List[str]]:
    reader = PdfReader(io.BytesIO(src_bytes))
    odd_writer, even_writer = PdfWriter(), PdfWriter()
    texts: List[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        texts.append(text)
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

    sku = None
    m2 = SKU_LINE_RE.search(text)
    if m2:
        sku = m2.group(1)

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
        "box_index", "total_boxes", "shipment_id", "fnsku", "asin", "sku", "tracking"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow(asdict(r))
    return buf.getvalue().encode("utf-8")


# ---------- UI ----------
st.set_page_config(page_title="FBA & Shipping Labels Splitter", page_icon="📦", layout="centered")
st.title("📦 FBA & Shipping Labels Splitter")
st.write("Upload the single alternating-labels PDF from Amazon (FBA label, then shipping label, etc.). We'll split it into two PDFs and generate a summary with **automatic packaging instructions**.")

# Load packaging data from Google Sheets
with st.spinner("Loading packaging data from Google Sheets..."):
    packaging_data = load_packaging_data_from_sheets()
    if packaging_data:
        st.success(f"✅ Loaded packaging data for {len(packaging_data)} SKUs from Google Sheets")
    else:
        st.warning("⚠️ Could not load packaging data. Packaging instructions will not be available.")


with st.expander("Options", expanded=False):
    assume_fba_is_odd = st.checkbox("Assume FBA labels are on odd pages (1,3,5,...) (FBA, then Shipping)", value=True)
    st.caption("Disable this if your input starts with a shipping label instead.")
    custom_title = st.text_input("Summary title", value="Amazon order")
    show_ids = st.multiselect("Show IDs in lines", options=["FNSKU", "ASIN", "Tracking", "Shipment ID"], default=["ASIN"])
    include_packaging = st.checkbox("Include packaging instructions in summary", value=True)
    st.caption("Enable to automatically add master carton or repacking instructions based on Google Sheets data")

uploaded = st.file_uploader("Upload combined PDF", type=["pdf"])

if uploaded is not None:
    data = uploaded.read()
    odd_bytes, even_bytes, texts = split_pdf_odd_even(data)

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

    def label_key(row):
        return row.sku or row.fnsku or row.asin or "(Unknown)"

    groups = []
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
    for (a, b, key, ids) in groups:
        if a == b:
            rng = f"Box {a}:"
        else:
            rng = f"Box {a}-{b}:"
        
        # Add packaging instructions if enabled
        if include_packaging and packaging_data:
            instruction_type, instruction_text = determine_packaging_instruction(key, packaging_data)
            suffix = f" / {ids}" if ids else ""
            
            # Add special notes for specific products
            special_note = ""
            if "Trout" in key and "50" in key:
                special_note = " Ensure that the carton picked up is the one with 90 units inside."
            elif "Chicken" in key and "50" in key:
                special_note = " Ensure that the carton picked up is the one with 90 units inside."
            
            lines.append(f"{rng} {key}{suffix} - {instruction_text}.{special_note}")
        else:
            suffix = f" / {ids}" if ids else ""
            lines.append(f"{rng} {key}{suffix}")
    
    text_summary = "\n".join(lines)

    st.subheader("Downloads")
    st.download_button("⬇️ Download FBA labels PDF", data=fba_pdf, file_name="fba_labels.pdf", mime="application/pdf")
    st.download_button("⬇️ Download Shipping labels PDF", data=ship_pdf, file_name="shipping_labels.pdf", mime="application/pdf")

    if summary_rows:
        csv_bytes = to_csv_bytes(summary_rows)
        st.download_button("⬇️ Download summary CSV", data=csv_bytes, file_name="labels_summary.csv", mime="text/csv")

        st.subheader("Text summary for HIVE")
        st.code(text_summary, language=None)
        st.download_button("⬇️ Download summary TXT", data=text_summary.encode("utf-8"), file_name="labels_summary.txt", mime="text/plain")

        # Show packaging breakdown
        if include_packaging and packaging_data:
            with st.expander("📦 Packaging Breakdown"):
                for (a, b, key, ids) in groups:
                    instruction_type, instruction_text = determine_packaging_instruction(key, packaging_data)
                    if a == b:
                        box_label = f"Box {a}"
                    else:
                        box_label = f"Boxes {a}-{b}"
                    
                    if instruction_type == "master":
                        st.success(f"✅ {box_label}: **{key}** → Use master carton")
                    elif instruction_type == "repack":
                        st.warning(f"📦 {box_label}: **{key}** → Repack into HIVE boxes")
                    else:
                        st.info(f"❓ {box_label}: **{key}** → {instruction_text}")

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
- **Packaging data**: Automatically loaded from your Google Sheet. Update the sheet and refresh the page to see changes.
        """
    )
    
    st.header("Packaging Rules")
    st.markdown(
        """
**Master Carton Used When:**
- Weight ≤ 10kg AND
- Fully enclosed in closed carton

**Repack into HIVE Boxes When:**
- Weight > 10kg OR
- Not fully enclosed
        """
    )
    
    if packaging_data:
        st.header("Loaded SKUs")
        st.caption(f"{len(packaging_data)} SKUs loaded from Google Sheets")
        with st.expander("View SKU List"):
            for sku in sorted(packaging_data.keys()):
                st.text(f"• {sku}")
