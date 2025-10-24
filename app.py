import streamlit as st
import os
from dotenv import load_dotenv
from llama_cloud.client import AsyncLlamaCloud
from llama_cloud.types import ClassifierRule, ClassifyParsingConfiguration, ParserLanguages
from llama_cloud_services.beta.classifier.client import ClassifyClient
from llama_cloud_services import LlamaExtract
from pydantic import BaseModel, Field

# --------------------------
# Load API key
# --------------------------
load_dotenv()
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

client = AsyncLlamaCloud(token=LLAMA_API_KEY)
project_id = "42da2f89-2702-426f-b41f-0440b3858bdd"
organization_id = "e7103cc5-2717-4a01-abc3-f7ea6fc579b9"
classify_client = ClassifyClient(client, project_id=project_id, organization_id=organization_id)
extractor = LlamaExtract()

# --------------------------
# Define Schemas
# --------------------------
class Invoice(BaseModel):
    invoice_number: str
    date: str
    total: float
    items: list[dict]

class Receipt(BaseModel):
    merchant: str
    date: str
    total: float
    items: list[dict]

class Resume(BaseModel):
    name: str
    email: str
    phone: str
    skills: list[str]
    experience: list[str]

class AadhaarCard(BaseModel):
    name: str
    aadhaar_number: str
    dob: str
    gender: str
    address: str

class PANCard(BaseModel):
    name: str
    pan_number: str
    dob: str
    father_name: str

class BankStatement(BaseModel):
    account_holder: str
    account_number: str
    bank_name: str
    transactions: list[dict]

class CoverLetter(BaseModel):
    candidate_name: str
    email: str
    company_name: str
    position: str
    summary: str

class LabReport(BaseModel):
    patient_name: str
    test_name: str
    result: str
    normal_range: str
    date: str

class Prescription(BaseModel):
    doctor_name: str
    patient_name: str
    medicines: list[str]
    dosage: str
    date: str

class MedicalRecord(BaseModel):
    patient_name: str
    diagnosis: str
    treatment: str
    doctor_name: str
    date: str

# --------------------------
# Create / Load Agents
# --------------------------
agent_definitions = {
    "invoice-extractor": Invoice,
    "receipt-extractor": Receipt,
    "resume-extractor": Resume,
    "aadhaar-extractor": AadhaarCard,
    "pan-extractor": PANCard,
    "bank-extractor": BankStatement,
    "coverletter-extractor": CoverLetter,
    "labreport-extractor": LabReport,
    "prescription-extractor": Prescription,
    "medicalrecord-extractor": MedicalRecord,
}

agents = {}
for agent_name, schema in agent_definitions.items():
    try:
        agents[agent_name] = extractor.get_agent(agent_name)
    except:
        agents[agent_name] = extractor.create_agent(agent_name, schema)

# --------------------------
# Streamlit App
# --------------------------
st.set_page_config(page_title="📄 Universal Document Extractor", page_icon="📄", layout="wide")
st.markdown("""
<div style="background-color:#f5f5f7; padding:25px; border-radius:15px; text-align:center;">
    <h1 style="color:#111111; font-family:sans-serif;">📄 Universal Document Classifier & Extractor</h1>
    <p style="color:#555555; font-size:16px;">Upload any document (PDF/DOCX) and get structured data instantly!</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drag & drop your document here", type=["pdf", "docx"], help="Max size: 200MB")

def render_cards(data: dict):
    """Render extracted data in Apple-style cards."""
    st.markdown("<div style='display:flex; gap:20px; flex-wrap:wrap;'>", unsafe_allow_html=True)
    for key, value in data.items():
        if isinstance(value, list):
            value = "<br>".join([str(v) for v in value])
        st.markdown(f"""
        <div style="flex:1; background-color:#e0f7fa; padding:20px; border-radius:12px; min-width:250px;">
            <h4 style="margin:0; color:#00796b;">{key.replace('_', ' ').title()}</h4>
            <p style="margin:5px 0; font-weight:bold;">{value}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("🔍 Classifying document...")
    rules = [
        ClassifierRule(type="invoice", description="Contains invoice number, date, and total."),
        ClassifierRule(type="receipt", description="Contains purchase info."),
        ClassifierRule(type="resume", description="Contains skills, education, and experience."),
        ClassifierRule(type="aadhaar", description="Contains 12-digit Aadhaar number and address."),
        ClassifierRule(type="pan", description="Contains PAN number and name."),
        ClassifierRule(type="bank", description="Contains transaction details and account number."),
    ]
    parsing = ClassifyParsingConfiguration(lang=ParserLanguages.EN, max_pages=5)

    results = classify_client.classify_file_paths(
        rules=rules,
        file_input_paths=[temp_file_path],
        parsing_configuration=parsing
    )

    item = results.items[0]
    if item.result is None:
        st.error("⚠️ Classification failed for this document.")
    else:
        doc_type = item.result.type
        confidence = item.result.confidence
        st.markdown(f"""
        <div style="background-color:#e0f7fa; padding:15px; border-radius:10px; border-left:6px solid #00acc1;">
            <h3 style="margin:0; color:#00796b;">✅ Document classified as <strong>{doc_type}</strong></h3>
            <p style="margin:0; color:#004d40;">Confidence: {confidence:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("**Reasoning:**", item.result.reasoning)

        st.info("⚡ Extracting data...")
        agent_key = f"{doc_type}-extractor"
        extract_agent = agents.get(agent_key)

        if extract_agent:
            extract_result = extract_agent.extract(temp_file_path)
            st.markdown(f"### 🗂 {doc_type.replace('-', ' ').title()} Data")
            render_cards(extract_result.data)
        else:
            st.warning(f"No extractor found for document type: {doc_type}")

    os.remove(temp_file_path)
