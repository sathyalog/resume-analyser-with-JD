from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii_presidio(text: str) -> str:
    # 1. Analyze text for entities (EMAIL_ADDRESS, PHONE_NUMBER, PERSON, LOCATION, etc.)
    results = analyzer.analyze(
        text=text,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION", "UK_NINO"],
        language="en"
    )
    
    # 2. Anonymize/Redact detected entities
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    return anonymized_result.text
