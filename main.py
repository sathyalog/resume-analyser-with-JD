import os
import json
from typing import Literal, Optional, TypedDict, Dict, Any, List

import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

# LangGraph & LangChain Imports
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

# Local Modules & Infrastructure Imports
from database import get_pinecone_index
from helpers import extract_github_handle
from prompts import load_prompt
from mcp_github import run_github_mcp
from PII_detection import redact_pii_presidio
from firecrawl_scraping import extract_jd_from_url

# Import Centralized Storage Functions & Clean Schemas
from core.storage import load_json_data, save_json_data

# from core.storage import load_json_data, save_json_data, JSON_FILES
from core.schemas import (
    SkillEntry,
    ProjectEntry,
    ChallengeEntry,
    ResponsibilityEntry,
    MiscContextEntry,
    ResumeAutoParseModel,
    ScreeningModel,
    ResumeStudioOutput,
    LinkedInStudioOutput,
    GitHubReadmeOutput
)

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
    index = get_pinecone_index()
    return index

pinecone_index = load_environment_variables()

# 2. Configure Streamlit Page Layout
st.set_page_config(
    page_title="AI Career & Skill Hub Engine",
    page_icon="🚀",
    layout="wide",
)

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    temperature=0,
    max_tokens=2500,
)

extracted_resume_text = ""
github_username = None

if "parsed_candidate_roles" not in st.session_state or not st.session_state.parsed_candidate_roles:
    st.session_state.parsed_candidate_roles = ["Lead AI Engineer", "Software Engineer", "General Professional Experience"]

with st.sidebar:
    st.title("Candidate Workspace")
    uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        extracted_resume_text = ""
        for page in pdf_reader.pages:
            extracted_resume_text += page.extract_text() or ""

        github_username = extract_github_handle(extracted_resume_text)

        st.divider()
        st.subheader("Candidate Profile")
        if github_username:
            st.success(f"**GitHub Handle:** @{github_username}")
            st.markdown(f"[View Profile](https://github.com/{github_username})")
        else:
            st.warning("No GitHub handle found in resume.")

# TOP-LEVEL NAVBAR NAVIGATION TABS
tab_analyser, tab_skill_studio, tab_interview = st.tabs([
    "📄 Resume Analyser", 
    "✍️ Skill Hub & Content Studio", 
    "🎯 AI Interview Copilot"
])

# ==============================================================================
# TAB 1: RESUME ANALYSER WITH JD MATCHING
# ==============================================================================
with tab_analyser:
    st.title("Resume Analyser with JD")
    st.caption("Upload your resume in sidebar and paste the Job description in the text box below")

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
        github_handle: Optional[str]
        github_mcp_output: Optional[str]
        pii_scrubbed: bool
        rejection_feedback: str
        critique: str
        reflection_count: int

    structured_model = llm.with_structured_output(ScreeningModel)

    @traceable(name="analyse_resume_with_jd")
    def AnalyseResumeWithJD(state: ScreeningState) -> ScreeningState:
        resume_text = state.get("resume_text", "")
        job_description = state.get("job_description", "")

        prompt = f"""
        You are an expert technical recruiter parsing a Candidate Resume and a Job Description.
        Extract candidate_name, company_name, job_title, candidate_experience, experience_required, and detected_roles.
        Extract required_skills and candidate_skills exhaustively. Normalize common tech names.
        
        Candidate Resume: {resume_text}
        Job Description: {job_description}
        """
        output: ScreeningModel = structured_model.invoke(prompt)

        if output.detected_roles:
            st.session_state.parsed_candidate_roles = list(dict.fromkeys(output.detected_roles + st.session_state.parsed_candidate_roles))

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
        return "ShortList" if (skill_match >= 0.50 and candidate_exp >= exp_required) else "Reject"

    @traceable(name="shortlist")
    def ShortList(state: ScreeningState) -> ScreeningState:
        st.success(f"Shortlisted for {state.get('job_title')} at {state.get('company_name')} - {state.get('candidate_name')}")
        return state

    @traceable(name="reject")
    def Reject(state: ScreeningState) -> ScreeningState:
        st.error(f"Rejected for {state.get('job_title')} at {state.get('company_name')} - {state.get('candidate_name')}")
        return state

    @traceable(name="scrub_resume_pii_node")
    def scrub_resume_pii_node(state: Dict[str, Any]) -> Dict[str, Any]:
        raw_resume_text = state.get("resume_text", "")
        return {"resume_text": redact_pii_presidio(raw_resume_text), "pii_scrubbed": True} if raw_resume_text else {"pii_scrubbed": False}

    # Graph Assembly
    builder = StateGraph(ScreeningState)
    builder.add_node("scrub_pii", scrub_resume_pii_node)
    builder.add_node("AnalyseResumeWithJD", AnalyseResumeWithJD)
    builder.add_node("ShortList", ShortList)
    builder.add_node("Reject", Reject)
    builder.add_node("GenerateRejectFeedback", GenerateRejectFeedback)
    builder.add_node("ReflectAndVerify", ReflectAndVerify)
    builder.add_node("FinalizeFeedback", FinalizeFeedbackNode)

    builder.add_edge(START, "scrub_pii")
    builder.add_edge("scrub_pii", "AnalyseResumeWithJD")
    builder.add_conditional_edges("AnalyseResumeWithJD", CheckCriteria)
    builder.add_edge("ShortList", END)
    builder.add_edge("Reject", "GenerateRejectFeedback")
    builder.add_edge("GenerateRejectFeedback", "ReflectAndVerify")
    builder.add_conditional_edges("ReflectAndVerify", ShouldContinueReflection, {"GenerateRejectFeedback": "GenerateRejectFeedback", "FinalizeFeedback": "FinalizeFeedback"})
    builder.add_edge("FinalizeFeedback", END)
    resume_analyser_graph = builder.compile()

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
                        final_jd_text = f"Job Title: {extracted_data.get('job_title')}\nOverview:\n{extracted_data.get('job_overview')}"
                        st.success("Fetched Job Description successfully!")
                    else:
                        st.error("Unable to extract JD automatically. Switch to 'Paste Text'.")
                        final_jd_text = job_description_input
            else:
                final_jd_text = job_description_input

            if final_jd_text:
                with st.spinner("Analyzing candidate against job description..."):
                    initial_state: ScreeningState = {
                        "resume_text": extracted_resume_text,
                        "job_description": final_jd_text,
                        "github_handle": github_username,
                        "reflection_count": 0,
                    }
                    final_state = resume_analyser_graph.invoke(initial_state)

                matched_skills_set = {s.lower() for s in final_state.get("matched_skills", [])}
                def render_badges(skills):
                    return " ".join([f'<span style="background-color: {"#2e7d32" if s.lower() in matched_skills_set else "#424242"}; color: white; padding: 3px 8px; border-radius: 12px; margin-right: 5px;">{s}</span>' for s in skills])

                with st.container(border=True):
                    st.caption("🏢 **COMPANY DETAILS (JOB POSTING)**")
                    c1, c2 = st.columns(2)
                    c1.metric("Company Name", final_state.get("company_name", "N/A"))
                    c2.metric("Role Title", final_state.get("job_title", "N/A"))
                    st.markdown("**Required Skills:** " + render_badges(final_state.get("required_skills", [])), unsafe_allow_html=True)

                with st.container(border=True):
                    st.caption("👤 **CANDIDATE DETAILS (RESUME)**")
                    cand1, cand2 = st.columns(2)
                    cand1.metric("Candidate Experience", f"{final_state.get('candidate_experience', 0)} Yrs")
                    cand2.metric("Skill Match Score", f"{(final_state.get('skill_match', 0.0) * 100):.2f}%")
                    st.markdown("**Resume Skills:** " + render_badges(final_state.get("candidate_skills", [])), unsafe_allow_html=True)

                if final_state.get("rejection_feedback"):
                    st.info(final_state.get("rejection_feedback"))

                st.divider()
                st.subheader("GitHub MCP Analysis")
                if github_username:
                    mcp_res = run_github_mcp(final_state)
                    st.markdown(mcp_res.get("github_mcp_output", "No GitHub data available."))

# ==============================================================================
# TAB 2: SKILL HUB & CONTENT GENERATOR STUDIO
# ==============================================================================
with tab_skill_studio:
    st.title("✍️ Skill Hub & Content Studio")
    st.caption("Update your central experience store and generate custom profile markdown across formats.")
    
    if extracted_resume_text:
        if st.button("⚡ Auto-Populate JSONs from Uploaded Resume", type="secondary"):
            with st.spinner("Extracting projects, certifications, categorized skills, and responsibilities..."):
                auto_parser = llm.with_structured_output(ResumeAutoParseModel)
                
                system_prompt = (
                    "Exhaustively parse the candidate resume text.\n"
                    "Extract:\n"
                    "1. roles: List of job title strings.\n"
                    "2. categorized_skills: Group skills under explicit resume categories (e.g., 'Agentic AI & Orchestration', 'Vector Search & Advanced RAG', 'Guardrails & Governance', 'Backend & Systems', 'Frontend').\n"
                    "3. projects: List of objects containing project_name, tech_stack, description.\n"
                    "4. responsibilities: List of objects containing role_title and responsibilities array.\n"
                    "5. misc: List of objects containing category (Certifications, Courses, etc.) and content."
                )
                
                parsed_res: ResumeAutoParseModel = auto_parser.invoke(
                    f"{system_prompt}\n\nResume Text:\n{extracted_resume_text}"
                )
                
                if parsed_res.roles:
                    st.session_state.parsed_candidate_roles = list(dict.fromkeys(parsed_res.roles + st.session_state.parsed_candidate_roles))

                    # 1. Overwrite/Populate Categorized Skills in skills.json
                if parsed_res.categorized_skills:
                    skills_path = os.path.join("data", "skills.json")
                    with open(skills_path, "w", encoding="utf-8") as f:
                        json.dump([], f)

                    for cat_group in parsed_res.categorized_skills:
                        save_json_data("skills", cat_group.model_dump())

                # 2. Populate projects.json
                for proj in parsed_res.projects:
                    save_json_data("projects", proj.model_dump())

                # 3. Populate responsibilities.json
                for resp in parsed_res.responsibilities:
                    save_json_data("responsibilities", {
                        "context_title": resp.role_title if resp.role_title else "General Experience",
                        "responsibilities": resp.responsibilities
                    })

                # 4. Populate misc.json
                for m in parsed_res.misc:
                    save_json_data("misc", m.model_dump())
                    
                st.session_state.resume_parsed = True
                st.success("Successfully populated projects.json, skills.json, responsibilities.json, and misc.json!")

    subtab_ingest, subtab_studio = st.tabs(["📥 Context Ingestion Engine", "📤 Profile Output Studio"])

    with subtab_ingest:
        col_proj, col_chal = st.columns(2)

        with col_proj:
            st.subheader("🛠️ Projects & Tech Stacks (`projects.json` and `skills.json`)")
            with st.form("form_projects"):
                p_name = st.text_input("Project Name / Ecosystem")
                p_tech = st.text_input("Tech Stack (comma separated)")
                p_desc = st.text_area("Detailed Architecture Description")
                
                if st.form_submit_button("Save Project & Skills"):
                    parsed_skills = [t.strip() for t in p_tech.split(",") if t.strip()]
                    
                    save_json_data("projects", {
                        "project_name": p_name, 
                        "tech_stack": parsed_skills, 
                        "description": p_desc
                    })
                    
                    save_json_data("skills", {
                        "category": f"Project Stack ({p_name})",
                        "skills": parsed_skills
                    })

                    st.success(f"Saved {p_name} to projects.json and updated skills.json!")

        with col_chal:
            st.subheader("🧩 Engineering Challenges (`challenges.json`)")
            with st.form("form_challenges"):
                raw_challenge_input = st.text_area(
                    "Describe your technical challenge in plain English:",
                    placeholder="e.g., We had severe database locks during peak sale hours due to poorly indexed queries. I optimized indexes and added Redis caching, dropping latency by 60%.",
                    height=180
                )
                
                if st.form_submit_button("Format with LLM & Save to challenges.json"):
                    if raw_challenge_input.strip():
                        with st.spinner("Structuring challenge into STAR format via LLM..."):
                            challenge_structurer = llm.with_structured_output(ChallengeEntry)
                            parsed_ch: ChallengeEntry = challenge_structurer.invoke(
                                f"Convert this raw engineering incident explanation into a structured STAR format with a title, scenario, solution, and impact:\n{raw_challenge_input}"
                            )
                            
                            save_json_data("challenges", parsed_ch.model_dump())
                            st.success(f"Formatted and saved challenge: '{parsed_ch.title}'!")
                    else:
                        st.warning("Please provide a description of the challenge.")

        st.divider()
        col_resp, col_misc = st.columns(2)

        with col_resp:
            st.subheader("📋 Day-to-Day Responsibilities (`responsibilities.json`)")
            with st.form("form_responsibilities"):
                selected_role = st.selectbox(
                    "Select Role / Context Title (Detected from Resume):", 
                    options=st.session_state.parsed_candidate_roles
                )
                r_items = st.text_area(
                    "Day-to-Day Responsibilities (one per line)", 
                    placeholder="Managed CI/CD deployments\nLed daily standups and code reviews\nOptimized PostgreSQL database queries", 
                    height=150
                )
                
                if st.form_submit_button("Save to responsibilities.json"):
                    parsed_responsibilities = [r.strip() for r in r_items.split("\n") if r.strip()]
                    save_json_data("responsibilities", {
                        "context_title": selected_role,
                        "responsibilities": parsed_responsibilities
                    })
                    st.success(f"Saved responsibilities for '{selected_role}' to responsibilities.json!")

        with col_misc:
            st.subheader("📝 Miscellaneous Credentials (`misc.json`)")
            with st.form("form_misc"):
                m_cat = st.selectbox(
                    "Category", 
                    ["Certifications", "Courses Learned", "Leadership & Mentorship", "Patents / Publications", "Other"]
                )
                m_content = st.text_area("Credential / Milestone Details")
                if st.form_submit_button("Save to misc.json"):
                    save_json_data("misc", {"category": m_cat, "content": m_content})
                    st.success("Saved entry to misc.json!")

    with subtab_studio:
        st.subheader("Target Profile Formatter")
        output_type = st.selectbox("Select Target Format", ["ATS Resume Content", "LinkedIn Summary", "GitHub Profile README"])

        has_skills = len(load_json_data("skills")) > 0
        has_projects = len(load_json_data("projects")) > 0
        resume_is_parsed = st.session_state.get("resume_parsed", False) or has_skills or has_projects

        if not resume_is_parsed:
            st.warning("⚠️ Please upload a resume or auto-populate JSONs before generating profile outputs.")

        if st.button("Generate Formatted Output", type="primary", disabled=not resume_is_parsed):
            context = {
                "skills": load_json_data("skills"),
                "projects": load_json_data("projects"),
                "responsibilities": load_json_data("responsibilities"),
                "challenges": load_json_data("challenges"),
                "misc": load_json_data("misc")
            }

            with st.spinner("Generating target layout..."):
                if output_type == "LinkedIn Summary":
                    raw_yaml_prompt = load_prompt("linkedin_prompts.yaml", "linkedin_summary_prompt")
                    
                    formatted_prompt = raw_yaml_prompt.format(
                        years_exp=14,
                        core_stack="React, Node.js, .NET, SQL",
                        cloud_exp=5,
                        cloud_platform="Azure",
                        context_json=json.dumps(context, indent=2)
                    )
                    
                    system_instruction = (
                        "You are an executive LinkedIn profile writer. "
                        "You MUST respond ONLY with a valid JSON object containing 3 keys: "
                        '"headline", "about_section", and "featured_hashtags". '
                        "Do NOT include markdown wrapping outside the JSON."
                    )
                    
                    try:
                        raw_response = llm.invoke([
                            SystemMessage(content=system_instruction),
                            HumanMessage(content=formatted_prompt)
                        ])
                        
                        raw_text = raw_response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                        parsed = json.loads(raw_text)
                        headline_text = parsed.get("headline", "")
                        about_text = parsed.get("about_section", "")
                        hashtags_list = parsed.get("featured_hashtags", [])
                        
                    except Exception:
                        headline_text = ""
                        about_text = raw_response.content.strip()
                        hashtags_list = []

                    about_text = about_text.replace("\\n", "\n").strip()

                    if headline_text:
                        st.markdown(f"**Headline:** `{headline_text}`")
                    
                    st.markdown("### LinkedIn About Section")
                    if about_text:
                        st.caption("Click the copy button in the top right corner of the block below to copy your summary:")
                        st.code(about_text, language="markdown")
                    else:
                        st.error("Could not generate summary. Please click 'Auto-Populate JSONs' above and try again.")

                    if hashtags_list:
                        formatted_tags = " ".join([f"#{t.replace('#', '').strip()}" for t in hashtags_list if t.strip()])
                        st.write(f"**Hashtags:** {formatted_tags}")

                elif output_type == "ATS Resume Content":
                    raw_yaml_prompt = load_prompt("resume_prompts.yaml", "resume_content_prompt")
                    formatted_prompt = raw_yaml_prompt.format(context_json=json.dumps(context, indent=2))
                    
                    try:
                        raw_response = llm.invoke([
                            SystemMessage(content='Respond ONLY with valid JSON containing keys: "professional_summary", "highlighted_bullets", and "technical_skills_formatted".'),
                            HumanMessage(content=formatted_prompt)
                        ])
                        raw_text = raw_response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                        parsed = json.loads(raw_text)
                        prof_summary = parsed.get("professional_summary", "")
                        bullets = parsed.get("highlighted_bullets", [])
                        skills_block = parsed.get("technical_skills_formatted", "")
                    except Exception:
                        prof_summary = raw_response.content.strip()
                        bullets = []
                        skills_block = ""

                    st.markdown("### Professional Summary")
                    st.write(prof_summary)
                    if bullets:
                        st.markdown("### Highlighted Bullet Points")
                        for bullet in bullets:
                            st.markdown(f"* {bullet}")
                    if skills_block:
                        st.markdown("### Technical Skills Block")
                        st.code(skills_block, language="markdown")

                elif output_type == "GitHub Profile README":
                    raw_yaml_prompt = load_prompt("github_prompts.yaml", "github_readme_prompt")
                    formatted_prompt = raw_yaml_prompt.format(
                        candidate_name="Sathya Vakacharla",
                        github_handle=github_username if github_username else "sathyalog",
                        context_json=json.dumps(context, indent=2)
                    )
                    
                    try:
                        raw_response = llm.invoke([
                            SystemMessage(content='Respond ONLY with a valid JSON object containing the key "markdown_readme".'),
                            HumanMessage(content=formatted_prompt)
                        ])
                        raw_text = raw_response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                        parsed = json.loads(raw_text)
                        readme_md = parsed.get("markdown_readme", "")
                    except Exception:
                        readme_md = raw_response.content.strip()

                    readme_md = readme_md.replace("\\n", "\n").strip()

                    st.markdown("### Rendered README Preview")
                    st.markdown(readme_md)
                    st.divider()
                    st.subheader("Raw Markdown Code (Copyable)")
                    st.code(readme_md, language="markdown")


# ==============================================================================
# TAB 3: AI INTERVIEW COPILOT
# ==============================================================================
with tab_interview:
    st.title("🎯 AI Interview Preparation Copilot")
    st.caption("Generate interview answers or glance over your saved day-to-day responsibilities.")

    prep_mode = st.radio(
        "Select Preparation Section:",
        ["📋 Responsibilities Quick-Glance", "Tell Me About Yourself", "STAR Method Technical Challenges", "Tech Stack Architecture Cheatsheet"],
        horizontal=True
    )

    if prep_mode == "📋 Responsibilities Quick-Glance":
        st.subheader("Day-to-Day Responsibilities Glance")
        resp_data = load_json_data("responsibilities")
        
        if resp_data:
            for idx, entry in enumerate(resp_data, 1):
                with st.expander(f"🔹 {entry.get('context_title', f'Role Context #{idx}')}", expanded=True):
                    responsibilities_list = entry.get("responsibilities", [])
                    for item in responsibilities_list:
                        st.markdown(f"* {item}")
        else:
            st.info("No responsibilities added yet. Add them in Tab 2 under the 'Context Ingestion Engine'.")

    else:
        if st.button("Generate Interview Strategy Output", type="primary"):
            context = {
                "skills": load_json_data("skills"),
                "projects": load_json_data("projects"),
                "responsibilities": load_json_data("responsibilities"),
                "challenges": load_json_data("challenges"),
                "misc": load_json_data("misc")
            }

            with st.spinner("Parsing career history and drafting responses..."):
                if prep_mode == "Tell Me About Yourself":
                    prompt = f"Context: {json.dumps(context)}. Generate a 2-minute elevator pitch highlighting core technical leadership."
                    ans = llm.invoke([HumanMessage(content=prompt)])
                    st.markdown(ans.content)

                elif prep_mode == "STAR Method Technical Challenges":
                    prompt = f"Context: {json.dumps(context['challenges'])}. Parse each challenge and convert into a clear STAR response."
                    ans = llm.invoke([HumanMessage(content=prompt)])
                    st.markdown(ans.content)

                elif prep_mode == "Tech Stack Architecture Cheatsheet":
                    prompt = f"Context: {json.dumps(context)}. Build an architectural interview cheatsheet covering system design decisions."
                    ans = llm.invoke([HumanMessage(content=prompt)])
                    st.markdown(ans.content)
