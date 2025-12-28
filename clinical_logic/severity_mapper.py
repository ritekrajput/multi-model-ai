def map_severity(score):
    """
    score: float (1–10)
    """

    if score <= 3:
        return {
            "level": "Early",
            "message": "Mild depressive indicators detected",
            "action": "Self-care and monitoring recommended"
        }

    elif score <= 6:
        return {
            "level": "Moderate",
            "message": "Moderate depressive symptoms detected",
            "action": "Professional consultation advised"
        }

    else:
        return {
            "level": "Severe",
            "message": "Severe depressive symptoms detected",
            "action": "Immediate professional help recommended"
        }
