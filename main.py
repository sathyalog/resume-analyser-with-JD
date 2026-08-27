import os
from typing import Literal, Optional, TypedDict, Dict, Any, List
from database import get_pinecone_index
from dotenv import load_dotenv
from helpers import extract_github_handle

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from mcp_github import run_github_mcp
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


# 1. Load Environment Variables & Cached Index
@traceable(name="load_environment_variables")
def load_environment_variables():
    load_dotenv(override=True)
    # Retrieves cached Pinecone index (runs create check only ONCE on app startup)
    index = get_pinecone_index()
    return index


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

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=1000,
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


# 3. Enhanced Pydantic Model for Robust Extraction
class ScreeningModel(BaseModel):
    company_name: str = Field(description="Name of the hiring company.")
    candidate_name: str = Field(description="Full candidate name from top of resume.")
    job_title: str = Field(description="Job title mentioned in JD.")
    candidate_experience: float = Field(
        description="Total candidate professional experience in years."
    )
    experience_required: Optional[float] = Field(
        default=8.0, description="Required years of experience."
    )
    skill_match: Optional[float] = Field(
        default=0.0,
        description="Skill match score calculated as ratio of matched skills.",
    )
    required_skills: list[str] = Field(
        default_factory=list, description="Key technical skills, frameworks, languages, and platforms required by JD."
    )
    candidate_skills: list[str] = Field(
        default_factory=list, description="Every technical skill, tool, language, platform, or framework mentioned across the entire candidate resume."
    )
    matched_skills: list[str] = Field(
        default_factory=list,
        description="Intersection of required_skills found in candidate_skills.",
    )


# 4. TypedDict State
class ScreeningState(TypedDict, total=False):
    company_name: Optional[str]
    candidate_name: Optional[str]
    job_title: Optional[str]
    candidate_experience: Optional[float]
    experience_required: Optional[float]
    skill_match: Optional[float]
    required_skills: List[str]
    candidate_skills: List[str]
    matched_skills: List[str]
    resume_text: Optional[str]
    job_description: Optional[str]
    email: Optional[str]
    github_handle: Optional[str]
    github_mcp_output: Optional[str]
    pii_scrubbed: bool
    evaluation_result: Dict[str, Any]
    rejection_feedback: str
    critique: str
    reflection_count: int


structured_model = llm.with_structured_output(ScreeningModel)


# 5. Node definitions with safe state & normalized skill matching
@traceable(name="analyse_resume_with_jd")
def AnalyseResumeWithJD(state: ScreeningState) -> ScreeningState:
    resume_text = state.get("resume_text", "")
    job_description = state.get("job_description", "")

    prompt = f"""
    You are an expert technical recruiter parsing a Candidate Resume and a Job Description.

    CRITICAL INSTRUCTION FOR SKILL EXTRACTION:
    1. Extract `candidate_name`, `company_name`, `job_title`, `candidate_experience`, and `experience_required`.
    2. `required_skills`: List key technical skills, languages, tools, and platforms required in the Job Description.
    3. `candidate_skills`: Exhaustively list all technical skills, frameworks, libraries, tools, and platforms mentioned anywhere in the resume (including work history and project descriptions). DO NOT return an empty list if technical terms exist in the resume text.
    4. Normalize common tech names (e.g., 'React.JS' -> 'React', 'AWS' -> 'Amazon Web Services').

    Candidate Resume Text:
    {resume_text}

    Job Description Text:
    {job_description}
    """
    output: ScreeningModel = structured_model.invoke(prompt)

    # Normalized Case-Insensitive Skill Matching Logic
    req_skills = output.required_skills or []
    cand_skills = output.candidate_skills or []

    req_set_lower = {s.strip().lower() for s in req_skills if s.strip()}
    cand_set_lower = {s.strip().lower() for s in cand_skills if s.strip()}

    matched_set_lower = req_set_lower.intersection(cand_set_lower)
    exact_matched = [s for s in req_skills if s.strip().lower() in matched_set_lower]

    exact_score = len(matched_set_lower) / len(req_set_lower) if len(req_set_lower) > 0 else 0.0

    return {
        "company_name": output.company_name,
        "candidate_name": output.candidate_name,
        "skill_match": exact_score,
        "candidate_experience": output.candidate_experience,
        "experience_required": output.experience_required,
        "job_title": output.job_title,
        "required_skills": req_skills,
        "candidate_skills": cand_skills,
        "matched_skills": exact_matched,
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
        cleaned_text = redact_pii_presidio(raw_resume_text)
        return {"resume_text": cleaned_text, "pii_scrubbed": True}

    return {"pii_scrubbed": False}


# 6. Graph Builder
builder = StateGraph(ScreeningState)

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

builder.add_edge("ShortList", END)

# Rejection Reflection Pipeline
builder.add_edge("Reject", "GenerateRejectFeedback")
builder.add_edge("GenerateRejectFeedback", "ReflectAndVerify")

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

# 7. Action Button & Execution Trigger
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
                        "⚠️ Unable to extract JD automatically. Please switch input mode to 'Paste Text' and paste the Job Description manually."
                    )
                    final_jd_text = job_description_input
        else:
            final_jd_text = job_description_input

        if final_jd_text:
            with st.spinner("Analyzing resume against job description..."):
                initial_state: ScreeningState = {
                    "resume_text": extracted_resume_text,
                    "job_description": final_jd_text,
                    "github_handle": github_username,
                    "reflection_count": 0,
                }

                final_state = resume_analyser_graph.invoke(initial_state)

            st.subheader("Analysis Breakdown")

            def render_skill_badges(skills_list, matched_set):
                badges = []
                for skill in skills_list:
                    if skill.lower() in matched_set:
                        badges.append(
                            f'<span style="background-color: #2e7d32; color: white; padding: 3px 8px; border-radius: 12px; font-weight: bold; margin-right: 5px; display: inline-block; margin-bottom: 5px;">✓ {skill}</span>'
                        )
                    else:
                        badges.append(
                            f'<span style="background-color: #424242; color: white; padding: 3px 8px; border-radius: 12px; margin-right: 5px; display: inline-block; margin-bottom: 5px;">{skill}</span>'
                        )
                return " ".join(badges)

            matched_skills_set = {
                s.lower() for s in final_state.get("matched_skills", [])
            }

            # Section 1: Company Details Container
            with st.container(border=True):
                st.caption("🏢 **COMPANY DETAILS (JOB POSTING)**")
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.metric("Company Name", final_state.get("company_name", "N/A"))
                with c_col2:
                    st.metric("Role Title", final_state.get("job_title", "N/A"))
                st.metric(
                    "Required Exp.", f"{final_state.get('experience_required', 0)} Yrs"
                )

                st.markdown("**Required Skills:**")
                req_skills = final_state.get("required_skills", [])
                if req_skills:
                    st.markdown(
                        render_skill_badges(req_skills, matched_skills_set),
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("None specified")

            # Section 2: Candidate Details Container
            with st.container(border=True):
                st.caption("👤 **CANDIDATE DETAILS (RESUME)**")
                cand_col1, cand_col2 = st.columns(2)
                st.metric("Candidate Name", final_state.get("candidate_name", "N/A"))
                with cand_col1:
                    st.metric(
                        "Candidate Exp.",
                        f"{final_state.get('candidate_experience', 0)} Yrs",
                    )
                with cand_col2:
                    exact_pct = final_state.get("skill_match", 0.0) * 100
                    st.metric("Skill Match Score", f"{exact_pct:.2f}%")

                st.markdown("**Candidate Resume Skills:**")
                cand_skills = final_state.get("candidate_skills", [])
                if cand_skills:
                    st.markdown(
                        render_skill_badges(cand_skills, matched_skills_set),
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("None extracted")

            feedback_output = final_state.get("rejection_feedback")
            if feedback_output:
                st.divider()
                st.subheader("💡 Candidate Career Coaching & Skill Gap Analysis")
                st.info(feedback_output)

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
