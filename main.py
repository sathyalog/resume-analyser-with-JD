from langgraph.graph import START, END, StateGraph
from langchain_openrouter import ChatOpenRouter
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pinecone import Pinecone
import streamlit as st
from database import create_index
import os

# 1. Load Environment Variables
load_dotenv(override=True)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
connect_database = create_index()
print(pc.list_indexes())

# 2. Configure Streamlit Page Layout
st.set_page_config(
    page_title="Resume Analyser with JD",
    page_icon="👋",
    layout="centered",
)

st.title("Resume Analyser with JD")
if "documents" not in st.session_state:
    st.session_state.documents = None
st.caption(
    "Upload your resume in sidebar and paste the Job description in the text box below"
)

llm = ChatOpenRouter(model="gpt-3.5-turbo")

st.text_area("Job Description", key="job_description", height=200)

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    if uploaded_file is not None:
        st.session_state.documents = uploaded_file
        st.write("File uploaded successfully!")

# 3. Pydantic Model for Structured Output
class ScreeningModel(BaseModel):
    candidate_name: str = Field(description="Name of the candidate")
    job_title: str = Field(description="Job title mentioned in job description")
    candidate_experience: float = Field(description="Working experience of the candidate as per the resume in years")
    experience_required: float = Field(description="Working experience required for the job as per the job description in years")
    skill_match: float = Field(description="Skill match score of the candidate. Value must be between 0 and 1", ge=0, le=1)

# 4. Corrected TypedDict (Pydantic Field() is removed from TypedDict)
class ScreeningState(TypedDict, total=False):
    candidate_name: Optional[str]
    job_title: Optional[str]
    candidate_experience: Optional[float]
    experience_required: Optional[float]
    skill_match: Optional[float]
    resume_text: Optional[str]
    job_description: Optional[str]
    email: Optional[str]


structured_model = llm.with_structured_output(ScreeningModel)

#5. Node definitions with safe state
def AnalyseResumeWithJD(state: ScreeningState) -> ScreeningState:
    resume_text = state["resume_text"]
    job_description = state["job_description"]

    prompt = f"""Analyze the provided resume text and job description to extract the candidate name and total years of experience from resume, and extract the job title and required years of experience from the job description. Compare the candidate's skills with the job requirements and compute a skill_match score as a float value between 0.0 and 1.0, where 0.0 indicates no relevant skills match the job description and 1.0 indicates strong alignment with most required skills, prioritizing skill relevance over job title.
    
    Resume Text:
    {resume_text}
    
    Job Description:
    {job_description}
    """
    output: ScreeningModel = structured_model.invoke(prompt)

    return {
        'candidate_name': output.candidate_name, 
        "skill_match": output.skill_match, 
        "candidate_experience": output.candidate_experience, 
        "experience_required": output.experience_required, 
        "job_title": output.job_title
    }

def CheckCriteria(state: ScreeningState) -> Literal["ShortList", "Reject"]:
    skill_match = state.get("skill_match", 0.0)
    candidate_exp = state.get("candidate_experience", 0.0)
    exp_required = state.get("experience_required", 0.0)

    if skill_match >= 0.50 and candidate_exp >= exp_required:
        return "ShortList"
    else:
        return "Reject"

def ShortList(state: ScreeningState) -> ScreeningState:
    candidate_name = state.get('candidate_name', 'Candidate')
    job_title = state.get('job_title', 'Position')
    message = f"Shortlisted for {job_title} - {candidate_name}"
    st.success(message)
    return message

def Reject(state: ScreeningState) -> ScreeningState:
    candidate_name = state.get('candidate_name', 'Candidate')
    job_title = state.get('job_title', 'Position')
    message = f"Rejected for {job_title} - {candidate_name}"
    st.error(message)
    return message


# 6. Graph Builder
builder = StateGraph(ScreeningState)
builder.add_node('AnalyseResumeWithJD', AnalyseResumeWithJD)
builder.add_node('ShortList', ShortList)
builder.add_node('Reject', Reject)

builder.add_edge(START, 'AnalyseResumeWithJD')
builder.add_conditional_edges('AnalyseResumeWithJD', CheckCriteria)
builder.add_edge('ShortList', END)
builder.add_edge('Reject', END)

resume_analyser_graph = builder.compile()
