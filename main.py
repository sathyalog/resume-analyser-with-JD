import os
from typing import Literal, Optional, TypedDict, Dict, Any
from database import create_index
from dotenv import load_dotenv
from helpers import extract_github_handle

# from langchain_openrouter import ChatOpenRouter
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from mcp_github import run_github_mcp
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pypdf import PdfReader
from PII_detection import redact_pii_presidio
import streamlit as st

# Import Firecrawl extraction function from local file
from firecrawl_scraping import extract_jd_from_url

# Import Deep Agent Reflection Nodes
from deepagent_feedback import (
    GenerateRejectFeedback,
    ReflectAndVerify,
    ShouldContinueReflection,
    FinalizeFeedbackNode,
)


# 1. Load Environment Variables
@traceable(name="load_environment_variables")
def load_environment_variables():
    load_dotenv(override=True)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    connect_database = create_index()
    print(pc.list_indexes())


load_environment_variables()

# 2. Configure Streamlit Page Layout
st.set_page_config(
    page_title="Resume Analyser with JD",
    page_icon="🔍",
    layout="centered",
)

st.title("Resume Analyser with JD")
st.caption(
    "Upload your resume in sidebar and paste the Job description in the text box below"
)

# llm = ChatOpenRouter(model="gpt-3.5-turbo")
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # Active low-cost Haiku model
    temperature=0,
    max_tokens=500,
)

# Input Mode Toggle (URL vs Paste Text)
input_mode = st.radio(
    "Select Job Description Input Method:", ["URL Link", "Paste Text"], horizontal=True
)

job_description_input = ""

if input_mode == "URL Link":
    jd_url = st.text_input(
        "Job Posting URL",
        placeholder="https://www.linkedin.com/jobs/view/...",
        key="jd_url",
    )
    if jd_url:
        st.info("The Job Description will be parsed via Firecrawl when you analyze.")
else:
    job_description_input = st.text_area(
        "Job Description Text", key="job_description", height=200
    )

extracted_resume_text = ""
github_username = None

# Sidebar for file upload & GitHub handle detection
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Extract text directly upon upload to detect GitHub handle
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            extracted_resume_text += page.extract_text() or ""

        github_username = extract_github_handle(extracted_resume_text)

        st.divider()
        st.subheader("Candidate Links Detected")
        if github_username:
            st.success(f"**GitHub Handle:** @{github_username}")
            st.markdown(f"[View Profile](https://github.com/{github_username})")
        else:
            st.warning("No GitHub handle found in resume.")


# 3. Pydantic Model for Structured Output
class ScreeningModel(BaseModel):
    company_name: str = Field(
        description="Name of the hiring company. Infer from email domains, text headers, or context. DO NOT output '<UNKNOWN>'. Use 'Not Specified' if impossible to identify."
    )
    candidate_name: str = Field(
        description="Full name of the candidate extracted from the top of the resume. DO NOT output '<UNKNOWN>'."
    )
    job_title: str = Field(
        description="Job title mentioned in the job description. Infer from context if necessary. DO NOT output '<UNKNOWN>'."
    )
    candidate_experience: float = Field(
        description="Total working experience of the candidate in years based on resume dates."
    )
    experience_required: Optional[float] = Field(
        default=8.0,
        description="Years of experience required. If unstated, infer from title level (e.g., Lead = 8.0).",
    )
    skill_match: float = Field(
        description="Skill match score between candidate and job requirements from 0.0 to 1.0.",
        ge=0,
        le=1,
    )


# 4. TypedDict State
class ScreeningState(TypedDict, total=False):
    company_name: Optional[str]
    candidate_name: Optional[str]
    job_title: Optional[str]
    candidate_experience: Optional[float]
    experience_required: Optional[float]
    skill_match: Optional[float]
    resume_text: Optional[str]
    job_description: Optional[str]
    email: Optional[str]
    github_handle: Optional[str]
    github_mcp_output: Optional[str]
    pii_scrubbed: bool
    evaluation_result: Dict[str, Any]
    rejection_feedback: str
    critique: str
    reflection_count: int  # Loop counter for the deep agent


structured_model = llm.with_structured_output(ScreeningModel)


# 5. Node definitions with safe state
@traceable(name="analyse_resume_with_jd")
def AnalyseResumeWithJD(state: ScreeningState) -> ScreeningState:
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")

    prompt = f"""
    You are an expert technical recruiter parsing a Resume and a Job Description.

    CRITICAL INSTRUCTIONS:
    1. `candidate_name`: Extract the full name located at the top of the resume text.
    2. `company_name`: Identify the company hiring for this role. Check headers, job intro, copyright statements, or email/URL mentions. Return "Not Specified" if absent—NEVER output '<UNKNOWN>'.
    3. `job_title`: Extract the target role title (e.g., "Lead AI Engineer"). NEVER output '<UNKNOWN>'.
    4. `candidate_experience`: Calculate total professional years from the resume.
    5. `experience_required`: Extract required minimum experience. Infer based on seniority if unstated (Lead=8, Senior=5, Mid=3).
    6. `skill_match`: Compute technical skill overlap between 0.0 and 1.0.

    Candidate Resume Text:
    {resume_text}

    Job Description Text:
    {job_description}
    """
    output: ScreeningModel = structured_model.invoke(prompt)

    return {
        "company_name": output.company_name,
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
    candidate_name = state.get("candidate_name") or "Candidate"
    job_title = state.get("job_title") or "Position"
    company_name = state.get("company_name") or "Company"
    message = f"Shortlisted for {job_title} at {company_name} - {candidate_name}"
    st.success(message)
    return state


@traceable(name="reject")
def Reject(state: ScreeningState) -> ScreeningState:
    candidate_name = state.get("candidate_name", "Candidate")
    job_title = state.get("job_title", "Position")
    company_name = state.get("company_name", "Company")
    message = f"Rejected for {job_title} at {company_name} - {candidate_name}"
    st.error(message)
    return state


@traceable(name="scrub_resume_pii_node")
def scrub_resume_pii_node(state: Dict[str, Any]) -> Dict[str, Any]:
    raw_resume_text = state.get("resume_text", "")

    if raw_resume_text:
        # Scrub sensitive data before passing to OpenRouter/LLM nodes
        cleaned_text = redact_pii_presidio(raw_resume_text)
        return {"resume_text": cleaned_text, "pii_scrubbed": True}

    return {"pii_scrubbed": False}


# 6. Graph Builder with PII detection nodes and Deep agent reflection loop
builder = StateGraph(ScreeningState)

# Add Nodes
builder.add_node("scrub_pii", scrub_resume_pii_node)
builder.add_node("AnalyseResumeWithJD", AnalyseResumeWithJD)
builder.add_node("ShortList", ShortList)
builder.add_node("Reject", Reject)

# Deep Reflection Agent Nodes
builder.add_node("GenerateRejectFeedback", GenerateRejectFeedback)
builder.add_node("ReflectAndVerify", ReflectAndVerify)
builder.add_node("FinalizeFeedback", FinalizeFeedbackNode)

# Primary Edges
builder.add_edge(START, "scrub_pii")
builder.add_edge("scrub_pii", "AnalyseResumeWithJD")
builder.add_conditional_edges("AnalyseResumeWithJD", CheckCriteria)

# Shortlist Pathway
builder.add_edge("ShortList", END)

# Rejection Reflection Pipeline
builder.add_edge("Reject", "GenerateRejectFeedback")
builder.add_edge("GenerateRejectFeedback", "ReflectAndVerify")

# Multi-Turn Deep Reflection Routing Loop
builder.add_conditional_edges(
    "ReflectAndVerify",
    ShouldContinueReflection,
    {
        "GenerateRejectFeedback": "GenerateRejectFeedback",
        "FinalizeFeedback": "FinalizeFeedback",
    },
)

builder.add_edge("FinalizeFeedback", END)

resume_analyser_graph = builder.compile()

# 7. Action Button & Graph Execution Trigger
st.divider()
if st.button("Analyze Candidate", type="primary"):
    if not uploaded_file:
        st.warning("Please upload a resume in PDF format.")
    elif input_mode == "URL Link" and not st.session_state.get("jd_url", "").strip():
        st.warning("Please provide a valid Job Description URL.")
    elif input_mode == "Paste Text" and not job_description_input.strip():
        st.warning("Please paste a Job Description.")
    else:
        final_jd_text = ""
        # Parse URL via Firecrawl or use direct text
        if input_mode == "URL Link":
            target_url = st.session_state.get("jd_url").strip()
            with st.spinner("Extracting Job Description from URL..."):
                extracted_data = extract_jd_from_url(target_url)

                if extracted_data and extracted_data.get("job_overview"):
                    skills_str = ", ".join(extracted_data.get("required_skills", []))
                    final_jd_text = f"""
                    Job Title: {extracted_data.get('job_title', 'N/A')}
                    Required Experience: {extracted_data.get('years_experience_required', 0)} years
                    Required Skills: {skills_str}
                    
                    Overview:
                    {extracted_data.get('job_overview', '')}
                    """
                    st.success("Successfully fetched Job Description!")
                else:
                    st.error(
                        "⚠️ This web domain (e.g., LinkedIn/Glassdoor) blocks automated scraping. Please switch input mode to 'Paste Text' and paste the Job Description manually."
                    )
                    final_jd_text = job_description_input
        else:
            final_jd_text = job_description_input

        # Execute screening workflow if JD text is ready
        if final_jd_text:
            with st.spinner("Analyzing resume against job description..."):
                initial_state: ScreeningState = {
                    "resume_text": extracted_resume_text,
                    "job_description": final_jd_text,
                    "github_handle": github_username,
                    "reflection_count": 0,
                }

                # Run LangGraph pipeline
                final_state = resume_analyser_graph.invoke(initial_state)

            # Display Structured Metrics Breakdown
            st.subheader("Analysis Breakdown")

            # Section 1: Company Details Container
            with st.container(border=True):
                st.caption("🏢 **COMPANY DETAILS (JOB POSTING)**")
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.metric("Company Name", final_state.get("company_name", "N/A"))
                with c_col2:
                    st.metric("Role Title", final_state.get("job_title", "N/A"))
                st.metric(
                    "Required Exp.",
                    f"{final_state.get('experience_required', 0)} Yrs",
                )
            # Section 2: Candidate Details Container
            with st.container(border=True):
                st.caption("👤 **CANDIDATE DETAILS (RESUME)**")
                cand_col1, cand_col2 = st.columns([1, 1])
                st.metric("Candidate Name", final_state.get("candidate_name", "N/A"))
                with cand_col1:
                    st.metric(
                        "Candidate Exp.",
                        f"{final_state.get('candidate_experience', 0)} Yrs",
                    )
                with cand_col2:
                    st.metric(
                        "Skill Match Score",
                        f"{round(final_state.get('skill_match', 0.0) * 100, 1)}%",
                    )

            # Display Deep Agent Feedback if Rejected
            feedback_output = final_state.get("rejection_feedback")
            if feedback_output:
                st.divider()
                st.subheader("💡 Candidate Career Coaching & Skill Gap Analysis")
                st.info(feedback_output)

            # Section for GitHub MCP Data Execution
            st.divider()
            st.subheader("GitHub MCP Analysis")
            if github_username:
                mcp_res = run_github_mcp(final_state)
                mcp_result = mcp_res.get("github_mcp_output")
                if mcp_result:
                    st.markdown(mcp_result)
                else:
                    st.info("No GitHub data available.")
            else:
                st.warning("No GitHub handle found to query MCP server.")
