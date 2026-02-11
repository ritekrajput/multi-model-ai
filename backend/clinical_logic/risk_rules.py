def apply_risk_rules(score, patient_text=None, relative_text=None):
    """
    Adds flags based on score & context
    """

    flags = []

    if score >= 8:
        flags.append("HIGH_RISK")

    if relative_text and "withdraw" in relative_text.lower():
        flags.append("SOCIAL_WITHDRAWAL")

    if patient_text and "hopeless" in patient_text.lower():
        flags.append("NEGATIVE_COGNITION")

    return flags
