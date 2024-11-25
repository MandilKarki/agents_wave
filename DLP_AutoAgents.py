# Install necessary packages
!pip install --quiet -U langchain_openai langchain_core langgraph

# Import necessary modules
import os
import getpass

def _set_env(var: str, value: str = None):
    if value:
        os.environ[var] = value
    elif not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set API keys
_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")

# Set LangChain tracing (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "dlp-agent"

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define data models
from pydantic import BaseModel, Field
from typing import List, Optional

class Threat(BaseModel):
    indicator: str = Field(description="Indicator of compromise (IP, domain, etc.).")
    type: str = Field(description="Type of threat (e.g., malicious_ip, malicious_domain, phishing_email).")
    description: str = Field(default="", description="Description of the threat.")
    severity: str = Field(default="", description="Severity level (e.g., Low, Medium, High, Critical).")
    status: str = Field(default="Pending", description="Analysis status (Pending, Analyzed).")

class SecurityAnalyst(BaseModel):
    name: str = Field(description="Name of the security analyst.")
    role: str = Field(description="Role or job title.")
    expertise: str = Field(description="Specific expertise or specialization.")
    experience_years: int = Field(description="Number of years of experience in the field.")
    certifications: List[str] = Field(description="Relevant certifications (e.g., CISSP, CEH).")
    personality_traits: List[str] = Field(description="Personality traits that affect analysis style.")
    description: str = Field(description="Detailed description of the analyst's focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        certifications = ', '.join(self.certifications)
        traits = ', '.join(self.personality_traits)
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Expertise: {self.expertise}\n"
            f"Experience: {self.experience_years} years\n"
            f"Certifications: {certifications}\n"
            f"Personality Traits: {traits}\n"
            f"Description: {self.description}\n"
        )

class AnalystTeam(BaseModel):
    analysts: List[SecurityAnalyst] = Field(
        description="List of security analysts with their roles and expertise."
    )

# Define the ThreatList model
class ThreatList(BaseModel):
    threats: List[Threat] = Field(description="List of extracted threats.")

# Define state schema and helper functions
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import operator

def merge_threats(left: List[Threat] | None, right: List[Threat] | None) -> List[Threat]:
    if left is None:
        left = []
    if right is None:
        right = []
    # Merge without duplicates based on the indicator
    threats = {threat.indicator: threat for threat in left + right}
    return list(threats.values())

def merge_analysis_sections(left: List[str] | None, right: List[str] | None) -> List[str]:
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right

def merge_remediation_steps(left: List[str] | None, right: List[str] | None) -> List[str]:
    return merge_analysis_sections(left, right)

class DLPState(TypedDict):
    user_id: str
    messages: Annotated[List[AnyMessage], add_messages]
    incident_description: str
    threats: Annotated[List[Threat], merge_threats]
    human_feedback: Optional[str]
    analysts: List[SecurityAnalyst]
    analysis_sections: Annotated[List[str], merge_analysis_sections]
    remediation_steps: Annotated[List[str], merge_remediation_steps]
    final_report: str
    max_analysts: int

# System messages and instructions
threat_extraction_instructions = """
You are a cybersecurity assistant. Your task is to identify potential threats from the incident description provided.

Instructions:
1. Extract all possible indicators of compromise (IOCs) such as IP addresses, domains, file hashes, email addresses, etc.
2. For each IOC, determine its type (e.g., malicious_ip, malicious_domain, phishing_email, suspicious_file_hash).
3. Do not provide remediation steps at this stage.
4. Present the threats in a structured format as per the ThreatList model.
"""

analyst_generation_instructions = """
You are tasked with creating a team of AI security analyst personas for incident analysis. Follow these instructions carefully:

1. Review the incident description:
{incident_description}

2. Examine any editorial feedback provided to guide the creation of the analysts:
{human_feedback}

3. Determine the key areas of focus based on the incident details and feedback.

4. Create up to {max_analysts} analysts, each specializing in a different area relevant to the incident.

5. For each analyst, provide the following:
   - Name
   - Role or job title
   - Specific expertise or specialization
   - Number of years of experience in the field
   - Relevant certifications (e.g., CISSP, CEH)
   - Personality traits that affect their analysis style (e.g., analytical, cautious, innovative)
   - A detailed description of their focus, concerns, and motives

6. Ensure that the team is diverse in expertise and perspective to cover all aspects of the incident.
"""

threat_analysis_instructions_template = """
You are {analyst_name}, a {role} with expertise in {expertise}. You have {experience_years} years of experience and hold the following certifications: {certifications}. Your personality traits are: {personality_traits}.

Instructions:
1. Analyze the following threat: {threat_indicator} ({threat_type}).
2. Provide a detailed analysis report including:
   - Findings
   - Potential impact
   - Severity assessment (Low, Medium, High, Critical)
   - Any additional observations based on your expertise
3. Use your personality traits to influence your analysis style.
4. Do not provide remediation steps in this section.
"""

remediation_instructions_template = """
You are {analyst_name}, a {role} with expertise in {expertise}.

Instructions:
1. Based on your analysis of {threat_indicator} ({threat_type}), provide recommended remediation steps.
2. Be specific and consider best practices in cybersecurity.
3. Address any potential challenges in implementing the remediation.
"""

# Define functions and nodes
from langchain_core.messages import SystemMessage, HumanMessage

def extract_threats(state: DLPState):
    incident_description = state['incident_description']

    # Enforce structured output
    structured_llm = llm.with_structured_output(ThreatList)

    # System message
    system_message = SystemMessage(content=threat_extraction_instructions)

    # Human message
    human_message = HumanMessage(content=incident_description)

    # Invoke LLM
    response = structured_llm.invoke([system_message, human_message])

    # Extract threats from response
    extracted_threats = response.threats

    return {'threats': extracted_threats}

def create_analysts(state: DLPState):
    incident_description = state['incident_description']
    human_feedback = state.get('human_feedback', '')
    max_analysts = state.get('max_analysts', 3)

    # Enforce structured output
    structured_llm = llm.with_structured_output(AnalystTeam)

    # System message
    system_message_content = analyst_generation_instructions.format(
        incident_description=incident_description,
        human_feedback=human_feedback,
        max_analysts=max_analysts
    )

    # Generate analysts
    analyst_team = structured_llm.invoke([
        SystemMessage(content=system_message_content),
        HumanMessage(content="Generate the analyst personas.")
    ])

    # Write the list of analysts to state
    return {"analysts": analyst_team.analysts}

def collect_human_feedback(state: DLPState):
    # Placeholder for human feedback
    pass

def should_continue(state: DLPState):
    feedback = state.get('human_feedback', None)
    if feedback:
        return 'create_analysts'
    else:
        return 'analyze_threats'

def analyze_threats(state: DLPState):
    threats = state['threats']
    analysts = state['analysts']
    analysis_sections = []

    num_analysts = len(analysts)
    for i, threat in enumerate(threats):
        assignment = {
            'analyst': analysts[i % num_analysts],
            'threat': threat
        }
        # Process the threat directly
        result = process_threat(assignment)
        analysis_sections.extend(result.get('analysis_sections', []))
        # Update the threat in the list
        threats[i] = result.get('threat', threat)

    # Update state
    return {
        'threats': threats,
        'analysis_sections': analysis_sections
    }

def process_threat(state: DLPState):
    threat = state['threat']
    analyst = state['analyst']

    # Prepare system message with analyst persona
    certifications = ', '.join(analyst.certifications)
    personality_traits = ', '.join(analyst.personality_traits)
    system_message_content = threat_analysis_instructions_template.format(
        analyst_name=analyst.name,
        role=analyst.role,
        expertise=analyst.expertise,
        experience_years=analyst.experience_years,
        certifications=certifications,
        personality_traits=personality_traits,
        threat_indicator=threat.indicator,
        threat_type=threat.type
    )

    system_message = SystemMessage(content=system_message_content)

    # Generate analysis report
    analysis_report = llm.invoke([system_message])

    # Update threat description and severity
    threat.description = analysis_report.content
    threat.status = "Analyzed"

    # Extract severity from the analysis
    if "Critical" in analysis_report.content:
        threat.severity = "Critical"
    elif "High" in analysis_report.content:
        threat.severity = "High"
    elif "Medium" in analysis_report.content:
        threat.severity = "Medium"
    else:
        threat.severity = "Low"

    # Store the analysis section
    analysis_section = f"**Analyst:** {analyst.name}\n**Threat:** {threat.indicator} ({threat.type})\n{analysis_report.content}\n"
    return {'threat': threat, 'analysis_sections': [analysis_section]}

def suggest_remediation_steps(state: DLPState):
    threats = state['threats']
    analysts = state['analysts']
    remediation_steps = []

    num_analysts = len(analysts)
    for i, threat in enumerate(threats):
        analyst = analysts[i % num_analysts]
        system_message_content = remediation_instructions_template.format(
            analyst_name=analyst.name,
            role=analyst.role,
            expertise=analyst.expertise,
            threat_indicator=threat.indicator,
            threat_type=threat.type
        )
        system_message = SystemMessage(content=system_message_content)
        step = llm.invoke([system_message])
        remediation_steps.append(f"**{analyst.name} recommends:** {step.content}")

    return {'remediation_steps': remediation_steps}

def compile_incident_report(state: DLPState):
    analysis_sections = state.get('analysis_sections', [])
    remediation_steps = state.get('remediation_steps', [])

    report = "# Incident Report\n\n## Analysis\n\n"
    report += "\n".join(analysis_sections)
    report += "\n## Remediation Steps\n\n"
    report += "\n".join(remediation_steps)

    return {'final_report': report}

# Build the main graph
from langgraph.graph import StateGraph, START, END

builder = StateGraph(DLPState)
builder.add_node('extract_threats', extract_threats)
builder.add_node('create_analysts', create_analysts)
builder.add_node('collect_human_feedback', collect_human_feedback)
builder.add_node('analyze_threats', analyze_threats)
builder.add_node('suggest_remediation_steps', suggest_remediation_steps)
builder.add_node('compile_incident_report', compile_incident_report)

# Edges
builder.add_edge(START, 'extract_threats')
builder.add_edge('extract_threats', 'create_analysts')
builder.add_edge('create_analysts', 'collect_human_feedback')
builder.add_conditional_edges('collect_human_feedback', should_continue, ['create_analysts', 'analyze_threats'])
builder.add_edge('analyze_threats', 'suggest_remediation_steps')
builder.add_edge('suggest_remediation_steps', 'compile_incident_report')
builder.add_edge('compile_incident_report', END)

# Compile the graph with interruption before 'collect_human_feedback'
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

dlp_graph = builder.compile(checkpointer=memory, interrupt_before=['collect_human_feedback'])

# Example usage
from langchain_core.messages import HumanMessage

# Initial state
initial_state = {
    'user_id': 'analyst1',
    'incident_description': (
        'We have detected unusual activity from IP 203.0.113.42 accessing sensitive data. '
        'Additionally, several phishing emails were sent from compromised accounts to external recipients. '
        'Suspicious file hashes have been identified in our system.'
    ),
    'messages': [],
    'max_analysts': 3
}

# Config
config = {'configurable': {'thread_id': 'dlp_thread1'}}

# Execute the graph up to the interruption point
result_state = dlp_graph.invoke(initial_state, config)

# Review generated analysts
print("\nGenerated Analysts for Review:")
for analyst in result_state['analysts']:
    print(f"Name: {analyst.name}")
    print(f"Role: {analyst.role}")
    print(f"Expertise: {analyst.expertise}")
    print(f"Experience: {analyst.experience_years} years")
    print(f"Certifications: {', '.join(analyst.certifications)}")
    print(f"Personality Traits: {', '.join(analyst.personality_traits)}")
    print(f"Description: {analyst.description}")
    print("-" * 80)

# Provide human feedback (if any)
# If no human feedback is needed, set 'human_feedback' to None
dlp_graph.update_state(
    config,
    {'human_feedback': None},  # Set to None or provide feedback as a string
    as_node='collect_human_feedback'
)

# Resume execution from the interruption point
for _ in dlp_graph.stream(None, config, stream_mode='updates'):
    pass

# Get the final state
final_state = dlp_graph.get_state(config)[0]

# Access and print the final report
print("\nDLP Incident Report:")
print(final_state['final_report'])