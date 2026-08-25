from typing import Dict, Any
from langsmith import traceable
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field

# Initialize LLM
feedback_llm = ChatOpenRouter(model="gpt-3.5-turbo")

# Pydantic schemas for structured outputs
class FeedbackOutput(BaseModel):
    missing_skills: list[str] = Field(description="List of specific skills required in JD but missing in resume")
    suggestions: str = Field(description="Actionable advice on how the candidate can bridge these skill gaps")

class CritiqueOutput(BaseModel):
    is_valid_gap: bool = Field(description="True if the identified missing skills are genuinely absent from the resume")
    critique_notes: str = Field(description="Detailed verification notes on whether any candidate experience was overlooked")

@traceable(name="GenerateRejectFeedback")
def GenerateRejectFeedback(state: Dict[str, Any]) -> Dict[str, Any]:
    resume = state.get("resume_text", "")
    jd = state.get("job_description", "")
    critique = state.get("critique", "No previous critique.")
    
    prompt = f"""You are an expert technical career coach. Compare the candidate's resume with the job description.
    Identify missing core skills and provide constructive recommendations.
    
    Previous Reviewer Critique (if any):
    {critique}
    
    Resume Text:
    {resume}
    
    Job Description:
    {jd}
    """
    
    structured_llm = feedback_llm.with_structured_output(FeedbackOutput)
    res: FeedbackOutput = structured_llm.invoke(prompt)
    
    feedback_str = f"**Missing Skills:** {', '.join(res.missing_skills)}\n\n**Suggestions:** {res.suggestions}"
    
    return {
        "rejection_feedback": feedback_str,
        "reflection_count": state.get("reflection_count", 0) + 1
    }

@traceable(name="ReflectAndVerify")
def ReflectAndVerify(state: Dict[str, Any]) -> Dict[str, Any]:
    resume = state.get("resume_text", "")
    feedback = state.get("rejection_feedback", "")
    
    prompt = f"""You are a Senior Hiring Quality Auditor. Review the proposed rejection feedback against the resume.
    Verify whether the candidate truly lacks these skills, or if they used alternative terminology (e.g., 'Docker' vs 'Containerization').
    
    Proposed Feedback:
    {feedback}
    
    Resume Text:
    {resume}
    """
    
    structured_llm = feedback_llm.with_structured_output(CritiqueOutput)
    res: CritiqueOutput = structured_llm.invoke(prompt)
    
    critique_flag = "NO_GAPS_FOUND" if res.is_valid_gap else "MISIDENTIFIED_GAP"
    critique_text = f"{critique_flag}: {res.critique_notes}"
    
    return {"critique": critique_text}

@traceable(name="ShouldContinueReflection")
def ShouldContinueReflection(state: Dict[str, Any]) -> str:
    count = state.get("reflection_count", 0)
    critique = state.get("critique", "")
    
    # Exit loop if feedback is verified valid (NO_GAPS_FOUND) OR max reflection passes (>= 2) are met
    if "NO_GAPS_FOUND" in critique or count >= 2:
        return "FinalizeFeedback"
    
    return "GenerateRejectFeedback"

@traceable(name="FinalizeFeedbackNode")
def FinalizeFeedbackNode(state: Dict[str, Any]) -> Dict[str, Any]:
    # Formats final verified feedback output
    return {"rejection_feedback": state.get("rejection_feedback", "No specific feedback generated.")}
