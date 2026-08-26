from typing import Dict, Any
from langsmith import traceable
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

# 1. Initialize LLM with a valid model and sufficient max_tokens
feedback_llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",  # Active low-cost Haiku model
    temperature=0,
    max_tokens=1000
)

# 2. Pydantic schemas with safe defaults to prevent ValidationErrors
class FeedbackOutput(BaseModel):
    missing_skills: list[str] = Field(
        default_factory=list, 
        description="List of specific skills required in JD but missing in resume"
    )
    suggestions: str = Field(
        default="Focus on building hands-on projects related to core JD requirements.", 
        description="Actionable advice on how the candidate can bridge these skill gaps"
    )

class CritiqueOutput(BaseModel):
    is_valid_gap: bool = Field(
        default=True, 
        description="True if the identified missing skills are genuinely absent from the resume"
    )
    critique_notes: str = Field(
        default="Verified skill gaps against candidate resume.", 
        description="Detailed verification notes on whether any candidate experience was overlooked"
    )

@traceable(name="GenerateRejectFeedback")
def GenerateRejectFeedback(state: Dict[str, Any]) -> Dict[str, Any]:
    resume = state.get("resume_text", "")
    jd = state.get("job_description", "")
    critique = state.get("critique", "No previous critique.")
    
    # 3. Explicit prompt forcing both JSON fields to be returned
    prompt = f"""You are an expert technical career coach. Compare the candidate's resume with the job description.
    Identify missing core skills and provide constructive recommendations.
    
    Previous Reviewer Critique (if any):
    {critique}
    
    Resume Text:
    {resume}
    
    Job Description:
    {jd}
    
    IMPORTANT: You MUST populate BOTH 'missing_skills' (as a list) and 'suggestions' (as a string) in your output schema.
    """
    
    structured_llm = feedback_llm.with_structured_output(FeedbackOutput)
    res: FeedbackOutput = structured_llm.invoke(prompt)
    
    missing_str = ", ".join(res.missing_skills) if res.missing_skills else "No specific technical skills flagged."
    feedback_str = f"**Missing Skills:** {missing_str}\n\n**Suggestions:** {res.suggestions}"
    
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
    
    IMPORTANT: You MUST populate BOTH 'is_valid_gap' (boolean) and 'critique_notes' (string) in your output schema.
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
