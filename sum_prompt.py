class Summarization(dspy.Signature):
    """
    Summarize the provided incident details into a **structured summary**, ensuring each field is properly mapped to its relevant context.

    **Focus on:**
    1. **Incident Cause:** Explain what triggered the incident.
    2. **Incident Progression:** Outline the key steps taken to address it (processes, communications, escalations).
    3. **Data Impact:** Highlight what data was involved and any assets affected.
    4. **Incident Outcome:** Provide a high-level conclusion about the resolution.

    **Ensure the summary follows this structured format:**

    ### Incident Summary
    - **Date of Offense:** {date_of_offense}
    - **Data Deleted:** {data_deleted}
    - **Manager Description:** {manager_description}
    - **Rationale:** {rationale}
    - **Description of Exposed Data:** {description_of_exposed_data}
    - **Identity Type:** {identity_type}
    - **Incident Priority:** {incident_priority}
    - **Cease & Desist:** {cease_desist}
    - **Declared Incident Indicator:** {declared_incident_indicator}
    - **Highest Classification:** {highest_classification}
    - **Number of Records Affected:** {number_of_records}
    - **Financial Loss:** {financial_loss}
    - **Data Misuse:** {data_misuse}
    - **Reputational Loss:** {reputational_loss}
    - **Operations Disruption:** {operations_disruption}
    - **Incident Types and Themes:** {incident_types_and_themes}

    **Guidelines for the summary:**
    - Ensure **all fields** are integrated naturally into the summary.
    - Use **concise, clear language** while maintaining logical flow.
    - Structure the summary in **a way that allows analysts to easily interpret it**.
    - Highlight **any unusual or critical findings** where relevant.
    """

text = dspy.InputField(desc="Incident details to be summarized.")

summary = dspy.OutputField(desc="""
Generate a structured summary using the provided incident details. 
- Ensure **all fields** are integrated into the summary.
- Maintain a **logical flow**, clearly mapping each field to its relevant context.
- Structure the output as per the template provided.
""")

# Initialize the ChainOfThought module with the Summarization signature
summarizer = dspy.ChainOfThought(Summarization)
