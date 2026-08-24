import os
from typing import Literal, Optional, TypedDict
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pypdf import PdfReader
import streamlit as st
from database import create_index
from langsmith import traceable

# 1. Load Environment Variables
@traceable(name="load_environment_variables")
def load_environment_variables():
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
st.caption(
    "Upload your resume in sidebar and paste the Job description in the text box below"
)

llm = ChatOpenRouter(model="gpt-3.5-turbo")

# Input text area for Job Description
job_description_input = st.text_area(
    "Job Description", key="job_description", height=200
)

# Sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")


# 3. Pydantic Model for Structured Output
class ScreeningModel(BaseModel):
    candidate_name: str = Field(description="Name of the candidate")
    job_title: str = Field(
        description="Job title mentioned in job description"
    )
    candidate_experience: float = Field(
        description="Working experience of the candidate as per the resume in years"
    )
    experience_required: float = Field(
        description="Working experience required for the job as per the job description in years"
    )
    skill_match: float = Field(
        description="Skill match score of the candidate. Value must be between 0 and 1",
        ge=0,
        le=1,
    )


# 4. TypedDict State
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


# 5. Node definitions with safe state
@traceable(name="analyse_resume_with_jd")
def AnalyseResumeWithJD(state: ScreeningState) -> ScreeningState:
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")

    prompt = f"""Analyze the provided resume text and job description to extract the candidate name and total years of experience from resume, and extract the job title and required years of experience from the job description. Compare the candidate's skills with the job requirements and compute a skill_match score as a float value between 0.0 and 1.0, where 0.0 indicates no relevant skills match the job description and 1.0 indicates strong alignment with most required skills, prioritizing skill relevance over job title.
    
    Resume Text:
    {resume_text}
    
    Job Description:
    {job_description}
    """
    output: ScreeningModel = structured_model.invoke(prompt)

    return {
        "candidate_name": output.candidate_name,
        "skill_match": output.skill_match,
        "candidate_experience": output.candidate_experience,
        "experience_required": output.experience_required,
        "job_title": output.job_title,
    }

@traceable(name="check_criteria")
def CheckCriteria(state: ScreeningState) -> Literal["ShortList", "Reject"]:
    skill_match = state.get("skill_match", 0.0)
    candidate_exp = state.get("candidate_experience", 0.0)
    exp_required = state.get("experience_required", 0.0)

    if skill_match >= 0.50 and candidate_exp >= exp_required:
        return "ShortList"
    else:
        return "Reject"

@traceable(name="shortlist")
def ShortList(state: ScreeningState) -> ScreeningState:
    candidate_name = state.get("candidate_name", "Candidate")
    job_title = state.get("job_title", "Position")
    message = f"Shortlisted for {job_title} - {candidate_name}"
    st.success(message)
    return state

@traceable(name="reject")
def Reject(state: ScreeningState) -> ScreeningState:
    candidate_name = state.get("candidate_name", "Candidate")
    job_title = state.get("job_title", "Position")
    message = f"Rejected for {job_title} - {candidate_name}"
    st.error(message)
    return state


# 6. Graph Builder
builder = StateGraph(ScreeningState)
builder.add_node("AnalyseResumeWithJD", AnalyseResumeWithJD)
builder.add_node("ShortList", ShortList)
builder.add_node("Reject", Reject)

builder.add_edge(START, "AnalyseResumeWithJD")
builder.add_conditional_edges("AnalyseResumeWithJD", CheckCriteria)
builder.add_edge("ShortList", END)
builder.add_edge("Reject", END)

resume_analyser_graph = builder.compile()

# 7. Action Button & Graph Execution Trigger
st.divider()
if st.button("Analyze Candidate", type="primary"):
    if not uploaded_file:
        st.warning("Please upload a resume in PDF format.")
    elif not job_description_input.strip():
        st.warning("Please paste a Job Description.")
    else:
        with st.spinner("Analyzing resume against job description..."):
            # Extract plain text from PDF
            pdf_reader = PdfReader(uploaded_file)
            extracted_resume_text = ""
            for page in pdf_reader.pages:
                extracted_resume_text += page.extract_text() or ""

            # Prepare initial graph state
            initial_state: ScreeningState = {
                "resume_text": extracted_resume_text,
                "job_description": job_description_input,
            }

            # Run LangGraph pipeline
            final_state = resume_analyser_graph.invoke(initial_state)

            # Display Extracted Metrics Breakdown
            st.subheader("Analysis Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Candidate Experience",
                    f"{final_state.get('candidate_experience', 0)} Years",
                )
                st.metric(
                    "Skill Match Score",
                    f"{round(final_state.get('skill_match', 0.0) * 100, 1)}%",
                )
            with col2:
                st.metric(
                    "Required Experience",
                    f"{final_state.get('experience_required', 0)} Years",
                )
                st.metric("Job Title Identified", final_state.get("job_title", "N/A"))
