from typing import List, Optional
from pydantic import BaseModel, Field

# ==============================================================================
# 1. LOCAL STORAGE & INGESTION SCHEMAS (data/*.json)
# ==============================================================================
class SkillEntry(BaseModel):
    category: str = Field(description="e.g., Agentic AI & Orchestration, Vector Search, Backend")
    skills: List[str] = Field(default_factory=list)

class ProjectEntry(BaseModel):
    project_name: str = Field(description="Title or identifier of the project")
    tech_stack: List[str] = Field(default_factory=list, description="Technologies used")
    description: str = Field(description="Architecture summary or achievements")

class ChallengeEntry(BaseModel):
    title: str = Field(description="Concise, technical title for the challenge")
    scenario: str = Field(description="Situation & Task background")
    solution: str = Field(description="Engineering action & solution")
    impact: str = Field(description="Quantifiable outcome achieved")

class ResponsibilityEntry(BaseModel):
    role_title: str = Field(description="Job title/role name")
    responsibilities: List[str] = Field(default_factory=list, description="List of day-to-day responsibilities")

class MiscContextEntry(BaseModel):
    category: str = Field(description="e.g., Certifications, Courses Learned, Leadership, Patents")
    content: str = Field(description="Details of credential/milestone")


# ==============================================================================
# 2. RESUME AUTO-PARSE MODELS
# ==============================================================================
class ResumeAutoParseModel(BaseModel):
    roles: List[str] = Field(default_factory=list, description="Job titles held")
    categorized_skills: List[SkillEntry] = Field(default_factory=list, description="Categorized skills list matching resume headings")
    projects: List[ProjectEntry] = Field(default_factory=list, description="Projects extracted from resume")
    responsibilities: List[ResponsibilityEntry] = Field(default_factory=list, description="Responsibilities per role")
    misc: List[MiscContextEntry] = Field(default_factory=list, description="Certifications, Courses, and Credentials")


# ==============================================================================
# 3. TAB 1: SCREENING & EVALUATION MODELS
# ==============================================================================
class ScreeningModel(BaseModel):
    company_name: str = Field(description="Name of the hiring company.")
    candidate_name: str = Field(description="Full candidate name from top of resume.")
    job_title: str = Field(description="Job title mentioned in JD.")
    candidate_experience: float = Field(description="Total candidate professional experience in years.")
    experience_required: Optional[float] = Field(default=8.0, description="Required years of experience.")
    skill_match: Optional[float] = Field(default=0.0, description="Skill match score calculated as ratio of matched skills.")
    required_skills: List[str] = Field(default_factory=list, description="Key technical skills required by JD.")
    candidate_skills: List[str] = Field(default_factory=list, description="Every technical skill mentioned across candidate resume.")
    matched_skills: List[str] = Field(default_factory=list, description="Intersection of required_skills found in candidate_skills.")
    detected_roles: List[str] = Field(default_factory=list, description="List of all work experience job titles found in resume.")


# ==============================================================================
# 4. TAB 2: PROFILE OUTPUT STUDIO SCHEMAS
# ==============================================================================
class ResumeStudioOutput(BaseModel):
    professional_summary: str = Field(default="", description="Executive summary highlighting technical expertise")
    highlighted_bullets: List[str] = Field(default_factory=list, description="ATS-optimized impact bullet points")
    technical_skills_formatted: str = Field(default="", description="Categorized skills block for resume header")

class LinkedInStudioOutput(BaseModel):
    headline: str = Field(default="", description="High-converting LinkedIn headline string.")
    about_section: str = Field(default="", description="Complete, rich text About section following layout contracts strictly.")
    featured_hashtags: List[str] = Field(default_factory=list, description="List of 5 key hashtags relevant to candidate profile.")

class GitHubReadmeOutput(BaseModel):
    markdown_readme: str = Field(default="", description="Complete valid Markdown content for GitHub profile README")
