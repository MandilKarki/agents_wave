{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X-j3ayHpTQEF",
        "outputId": "703880cd-7df2-4277-ef0a-4f878435860e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/50.4 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m50.4/50.4 kB\u001b[0m \u001b[31m1.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/409.5 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m409.5/409.5 kB\u001b[0m \u001b[31m12.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/125.1 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m125.1/125.1 kB\u001b[0m \u001b[31m9.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/1.2 MB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.2/1.2 MB\u001b[0m \u001b[31m45.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hOPENAI_API_KEY: ··········\n",
            "LANGCHAIN_API_KEY: ··········\n"
          ]
        }
      ],
      "source": [
        "# Install necessary packages\n",
        "!pip install --quiet -U langchain_openai langchain_core langgraph\n",
        "\n",
        "# Import necessary modules\n",
        "import os\n",
        "import getpass\n",
        "\n",
        "def _set_env(var: str, value: str = None):\n",
        "    # Function to set environment variables\n",
        "    if value:\n",
        "        os.environ[var] = value\n",
        "    elif not os.environ.get(var):\n",
        "        os.environ[var] = getpass.getpass(f\"{var}: \")\n",
        "\n",
        "# Set API keys\n",
        "_set_env(\"OPENAI_API_KEY\")\n",
        "_set_env(\"LANGCHAIN_API_KEY\")\n",
        "\n",
        "# Set LangChain tracing (optional)\n",
        "os.environ[\"LANGCHAIN_TRACING_V2\"] = \"true\"\n",
        "os.environ[\"LANGCHAIN_PROJECT\"] = \"dlp-agent\"\n",
        "\n",
        "from langchain_openai import ChatOpenAI\n",
        "llm = ChatOpenAI(model=\"gpt-4\", temperature=0)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##DATA MODELS\n",
        "\n"
      ],
      "metadata": {
        "id": "vO3J9MUzTmR8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from pydantic import BaseModel, Field\n",
        "from typing import List, Optional\n",
        "\n",
        "class Threat(BaseModel):\n",
        "    # Represents a cybersecurity threat\n",
        "    indicator: str = Field(description=\"Indicator of compromise (IP, domain, etc.).\")\n",
        "    type: str = Field(description=\"Type of threat (e.g., malicious_ip, malicious_domain, phishing_email).\")\n",
        "    description: str = Field(default=\"\", description=\"Description of the threat.\")\n",
        "    severity: str = Field(default=\"\", description=\"Severity level (e.g., Low, Medium, High, Critical).\")\n",
        "    status: str = Field(default=\"Pending\", description=\"Analysis status (Pending, Analyzed).\")\n",
        "\n",
        "class SecurityAnalyst(BaseModel):\n",
        "    # Represents a security analyst persona\n",
        "    name: str = Field(description=\"Name of the security analyst.\")\n",
        "    role: str = Field(description=\"Role or job title.\")\n",
        "    expertise: str = Field(description=\"Specific expertise or specialization.\")\n",
        "    experience_years: int = Field(description=\"Number of years of experience in the field.\")\n",
        "    certifications: List[str] = Field(description=\"Relevant certifications (e.g., CISSP, CEH).\")\n",
        "    personality_traits: List[str] = Field(description=\"Personality traits that affect analysis style.\")\n",
        "    description: str = Field(description=\"Detailed description of the analyst's focus, concerns, and motives.\")\n",
        "\n",
        "    @property\n",
        "    def persona(self) -> str:\n",
        "        # Returns a formatted string representing the analyst's persona\n",
        "        certifications = ', '.join(self.certifications)\n",
        "        traits = ', '.join(self.personality_traits)\n",
        "        return (\n",
        "            f\"Name: {self.name}\\n\"\n",
        "            f\"Role: {self.role}\\n\"\n",
        "            f\"Expertise: {self.expertise}\\n\"\n",
        "            f\"Experience: {self.experience_years} years\\n\"\n",
        "            f\"Certifications: {certifications}\\n\"\n",
        "            f\"Personality Traits: {traits}\\n\"\n",
        "            f\"Description: {self.description}\\n\"\n",
        "        )\n",
        "\n",
        "class AnalystTeam(BaseModel):\n",
        "    # Represents a team of security analysts\n",
        "    analysts: List[SecurityAnalyst] = Field(description=\"List of security analysts with their roles and expertise.\")\n",
        "\n",
        "class ThreatList(BaseModel):\n",
        "    # Represents a list of threats\n",
        "    threats: List[Threat] = Field(description=\"List of extracted threats.\")\n"
      ],
      "metadata": {
        "id": "stmpozU0TSBB"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##State Schema and Helper Functions"
      ],
      "metadata": {
        "id": "J5ls1F8QUAix"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from typing import TypedDict, Annotated\n",
        "from langchain_core.messages import AnyMessage\n",
        "from langgraph.graph.message import add_messages\n",
        "import operator\n",
        "\n",
        "def merge_threats(left: List[Threat] | None, right: List[Threat] | None) -> List[Threat]:\n",
        "    # Merges two lists of threats without duplicates based on the indicator\n",
        "    if left is None:\n",
        "        left = []\n",
        "    if right is None:\n",
        "        right = []\n",
        "    threats = {threat.indicator: threat for threat in left + right}\n",
        "    return list(threats.values())\n",
        "\n",
        "def merge_analysis_sections(left: List[str] | None, right: List[str] | None) -> List[str]:\n",
        "    # Merges two lists of analysis sections\n",
        "    if left is None:\n",
        "        left = []\n",
        "    if right is None:\n",
        "        right = []\n",
        "    return left + right\n",
        "\n",
        "def merge_remediation_steps(left: List[str] | None, right: List[str] | None) -> List[str]:\n",
        "    # Reuses the merge_analysis_sections function for remediation steps\n",
        "    return merge_analysis_sections(left, right)\n",
        "\n",
        "class DLPState(TypedDict):\n",
        "    # Defines the state schema used throughout the execution\n",
        "    user_id: str\n",
        "    messages: Annotated[List[AnyMessage], add_messages]\n",
        "    incident_description: str\n",
        "    threats: Annotated[List[Threat], merge_threats]\n",
        "    human_feedback: Optional[str]\n",
        "    analysts: List[SecurityAnalyst]\n",
        "    analysis_sections: Annotated[List[str], merge_analysis_sections]\n",
        "    remediation_steps: Annotated[List[str], merge_remediation_steps]\n",
        "    final_report: str\n",
        "    max_analysts: int\n"
      ],
      "metadata": {
        "id": "BMRWFiZuT2OG"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##System Messages and Instructions"
      ],
      "metadata": {
        "id": "HOkzHMZvVUaQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "threat_extraction_instructions = \"\"\"\n",
        "You are a cybersecurity assistant. Your task is to identify potential threats from the incident description provided.\n",
        "\n",
        "Instructions:\n",
        "1. Extract all possible indicators of compromise (IOCs) such as IP addresses, domains, file hashes, email addresses, etc.\n",
        "2. For each IOC, determine its type (e.g., malicious_ip, malicious_domain, phishing_email, suspicious_file_hash).\n",
        "3. Do not provide remediation steps at this stage.\n",
        "4. Present the threats in a structured format as per the ThreatList model.\n",
        "\"\"\"\n",
        "\n",
        "analyst_generation_instructions = \"\"\"\n",
        "You are tasked with creating a team of AI security analyst personas for incident analysis. Follow these instructions carefully:\n",
        "\n",
        "1. Review the incident description:\n",
        "{incident_description}\n",
        "\n",
        "2. Examine any editorial feedback provided to guide the creation of the analysts:\n",
        "{human_feedback}\n",
        "\n",
        "3. Determine the key areas of focus based on the incident details and feedback.\n",
        "\n",
        "4. Create up to {max_analysts} analysts, each specializing in a different area relevant to the incident.\n",
        "\n",
        "5. For each analyst, provide the following:\n",
        "   - Name\n",
        "   - Role or job title\n",
        "   - Specific expertise or specialization\n",
        "   - Number of years of experience in the field\n",
        "   - Relevant certifications (e.g., CISSP, CEH)\n",
        "   - Personality traits that affect their analysis style (e.g., analytical, cautious, innovative)\n",
        "   - A detailed description of their focus, concerns, and motives\n",
        "\n",
        "6. Ensure that the team is diverse in expertise and perspective to cover all aspects of the incident.\n",
        "\"\"\"\n",
        "\n",
        "threat_analysis_instructions_template = \"\"\"\n",
        "You are {analyst_name}, a {role} with expertise in {expertise}. You have {experience_years} years of experience and hold the following certifications: {certifications}. Your personality traits are: {personality_traits}.\n",
        "\n",
        "Instructions:\n",
        "1. Analyze the following threat: {threat_indicator} ({threat_type}).\n",
        "2. Provide a detailed analysis report including:\n",
        "   - Findings\n",
        "   - Potential impact\n",
        "   - Severity assessment (Low, Medium, High, Critical)\n",
        "   - Any additional observations based on your expertise\n",
        "3. Use your personality traits to influence your analysis style.\n",
        "4. Do not provide remediation steps in this section.\n",
        "\"\"\"\n",
        "\n",
        "remediation_instructions_template = \"\"\"\n",
        "You are {analyst_name}, a {role} with expertise in {expertise}.\n",
        "\n",
        "Instructions:\n",
        "1. Based on your analysis of {threat_indicator} ({threat_type}), provide recommended remediation steps.\n",
        "2. Be specific and consider best practices in cybersecurity.\n",
        "3. Address any potential challenges in implementing the remediation.\n",
        "\"\"\"\n",
        "\n",
        "# Define functions and nodes\n",
        "from langchain_core.messages import SystemMessage, HumanMessage\n",
        "\n",
        "def extract_threats(state: DLPState):\n",
        "    incident_description = state['incident_description']\n",
        "\n",
        "    # Enforce structured output\n",
        "    structured_llm = llm.with_structured_output(ThreatList)\n",
        "\n",
        "    # System message\n",
        "    system_message = SystemMessage(content=threat_extraction_instructions)\n",
        "\n",
        "    # Human message\n",
        "    human_message = HumanMessage(content=incident_description)\n",
        "\n",
        "    # Invoke LLM\n",
        "    response = structured_llm.invoke([system_message, human_message])\n",
        "\n",
        "    # Extract threats from response\n",
        "    extracted_threats = response.threats\n",
        "\n",
        "    return {'threats': extracted_threats}"
      ],
      "metadata": {
        "id": "zXVZOTbMUGnE"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Create Analysts"
      ],
      "metadata": {
        "id": "dgpnwBQOWQfj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def create_analysts(state: DLPState):\n",
        "    incident_description = state['incident_description']\n",
        "    human_feedback = state.get('human_feedback', '')\n",
        "    max_analysts = state.get('max_analysts', 3)\n",
        "\n",
        "    # Enforce structured output\n",
        "    structured_llm = llm.with_structured_output(AnalystTeam)\n",
        "\n",
        "    # System message\n",
        "    system_message_content = analyst_generation_instructions.format(\n",
        "        incident_description=incident_description,\n",
        "        human_feedback=human_feedback,\n",
        "        max_analysts=max_analysts\n",
        "    )\n",
        "\n",
        "    # Generate analysts\n",
        "    analyst_team = structured_llm.invoke([\n",
        "        SystemMessage(content=system_message_content),\n",
        "        HumanMessage(content=\"Generate the analyst personas.\")\n",
        "    ])\n",
        "\n",
        "    # Write the list of analysts to state\n",
        "    return {\"analysts\": analyst_team.analysts}\n"
      ],
      "metadata": {
        "id": "neVk2zRnV-Jd"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Collect HumanFeedback"
      ],
      "metadata": {
        "id": "oy93EJ5yWjIE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def collect_human_feedback(state: DLPState):\n",
        "    # Placeholder for human feedback\n",
        "    pass\n"
      ],
      "metadata": {
        "id": "s5-90T5mWUXS"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Should Continue"
      ],
      "metadata": {
        "id": "oElQFW3UXmR2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def should_continue(state: DLPState):\n",
        "    feedback = state.get('human_feedback', None)\n",
        "    if feedback:\n",
        "        return 'create_analysts'\n",
        "    else:\n",
        "        return 'analyze_threats'\n"
      ],
      "metadata": {
        "id": "p76QA5bVXg9i"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Analyse Threats"
      ],
      "metadata": {
        "id": "yIMIJn8_Xtjj"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def analyze_threats(state: DLPState):\n",
        "    threats = state['threats']\n",
        "    analysts = state['analysts']\n",
        "    analysis_sections = []\n",
        "\n",
        "    num_analysts = len(analysts)\n",
        "    for i, threat in enumerate(threats):\n",
        "        assignment = {\n",
        "            'analyst': analysts[i % num_analysts],\n",
        "            'threat': threat\n",
        "        }\n",
        "        # Process the threat directly\n",
        "        result = process_threat(assignment)\n",
        "        analysis_sections.extend(result.get('analysis_sections', []))\n",
        "        # Update the threat in the list\n",
        "        threats[i] = result.get('threat', threat)\n",
        "\n",
        "    # Update state\n",
        "    return {\n",
        "        'threats': threats,\n",
        "        'analysis_sections': analysis_sections\n",
        "    }\n"
      ],
      "metadata": {
        "id": "npW3oAo3XrJM"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##process_threat"
      ],
      "metadata": {
        "id": "aYjAwSUnX5Vn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def process_threat(state: DLPState):\n",
        "    threat = state['threat']\n",
        "    analyst = state['analyst']\n",
        "\n",
        "    # Prepare system message with analyst persona\n",
        "    certifications = ', '.join(analyst.certifications)\n",
        "    personality_traits = ', '.join(analyst.personality_traits)\n",
        "    system_message_content = threat_analysis_instructions_template.format(\n",
        "        analyst_name=analyst.name,\n",
        "        role=analyst.role,\n",
        "        expertise=analyst.expertise,\n",
        "        experience_years=analyst.experience_years,\n",
        "        certifications=certifications,\n",
        "        personality_traits=personality_traits,\n",
        "        threat_indicator=threat.indicator,\n",
        "        threat_type=threat.type\n",
        "    )\n",
        "\n",
        "    system_message = SystemMessage(content=system_message_content)\n",
        "\n",
        "    # Generate analysis report\n",
        "    analysis_report = llm.invoke([system_message])\n",
        "\n",
        "    # Update threat description and severity\n",
        "    threat.description = analysis_report.content\n",
        "    threat.status = \"Analyzed\"\n",
        "\n",
        "    # Extract severity from the analysis\n",
        "    if \"Critical\" in analysis_report.content:\n",
        "        threat.severity = \"Critical\"\n",
        "    elif \"High\" in analysis_report.content:\n",
        "        threat.severity = \"High\"\n",
        "    elif \"Medium\" in analysis_report.content:\n",
        "        threat.severity = \"Medium\"\n",
        "    else:\n",
        "        threat.severity = \"Low\"\n",
        "\n",
        "    # Store the analysis section\n",
        "    analysis_section = f\"**Analyst:** {analyst.name}\\n**Threat:** {threat.indicator} ({threat.type})\\n{analysis_report.content}\\n\"\n",
        "    return {'threat': threat, 'analysis_sections': [analysis_section]}\n"
      ],
      "metadata": {
        "id": "qOpr9CoVXyoP"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##suggest_remediation_steps\n",
        "\n"
      ],
      "metadata": {
        "id": "KON1R6-OYM1Q"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def suggest_remediation_steps(state: DLPState):\n",
        "    threats = state['threats']\n",
        "    analysts = state['analysts']\n",
        "    remediation_steps = []\n",
        "\n",
        "    num_analysts = len(analysts)\n",
        "    for i, threat in enumerate(threats):\n",
        "        analyst = analysts[i % num_analysts]\n",
        "        system_message_content = remediation_instructions_template.format(\n",
        "            analyst_name=analyst.name,\n",
        "            role=analyst.role,\n",
        "            expertise=analyst.expertise,\n",
        "            threat_indicator=threat.indicator,\n",
        "            threat_type=threat.type\n",
        "        )\n",
        "        system_message = SystemMessage(content=system_message_content)\n",
        "        step = llm.invoke([system_message])\n",
        "        remediation_steps.append(f\"**{analyst.name} recommends:** {step.content}\")\n",
        "\n",
        "    return {'remediation_steps': remediation_steps}\n"
      ],
      "metadata": {
        "id": "JQLQR64pX862"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##compile_incident_report"
      ],
      "metadata": {
        "id": "Ra0QWeaYYdRy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def compile_incident_report(state: DLPState):\n",
        "    analysis_sections = state.get('analysis_sections', [])\n",
        "    remediation_steps = state.get('remediation_steps', [])\n",
        "\n",
        "    report = \"# Incident Report\\n\\n## Analysis\\n\\n\"\n",
        "    report += \"\\n\".join(analysis_sections)\n",
        "    report += \"\\n## Remediation Steps\\n\\n\"\n",
        "    report += \"\\n\".join(remediation_steps)\n",
        "\n",
        "    return {'final_report': report}\n"
      ],
      "metadata": {
        "id": "VZoL5CSaYWJB"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Graph Construction"
      ],
      "metadata": {
        "id": "vggE5BbEYox8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from langgraph.graph import StateGraph, START, END\n",
        "\n",
        "builder = StateGraph(DLPState)\n",
        "builder.add_node('extract_threats', extract_threats)\n",
        "builder.add_node('create_analysts', create_analysts)\n",
        "builder.add_node('collect_human_feedback', collect_human_feedback)\n",
        "builder.add_node('analyze_threats', analyze_threats)\n",
        "builder.add_node('suggest_remediation_steps', suggest_remediation_steps)\n",
        "builder.add_node('compile_incident_report', compile_incident_report)\n",
        "\n",
        "# Edges\n",
        "builder.add_edge(START, 'extract_threats')\n",
        "builder.add_edge('extract_threats', 'create_analysts')\n",
        "builder.add_edge('create_analysts', 'collect_human_feedback')\n",
        "builder.add_conditional_edges('collect_human_feedback', should_continue, ['create_analysts', 'analyze_threats'])\n",
        "builder.add_edge('analyze_threats', 'suggest_remediation_steps')\n",
        "builder.add_edge('suggest_remediation_steps', 'compile_incident_report')\n",
        "builder.add_edge('compile_incident_report', END)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kxc73CEUYjIm",
        "outputId": "ed9794a8-7b5b-40b1-cae4-cc02999e8bcf"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<langgraph.graph.state.StateGraph at 0x7ad005b12140>"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Execution Flow"
      ],
      "metadata": {
        "id": "rBgkbqIUZLoc"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from langgraph.checkpoint.memory import MemorySaver\n",
        "memory = MemorySaver()\n",
        "\n",
        "dlp_graph = builder.compile(checkpointer=memory, interrupt_before=['collect_human_feedback'])\n"
      ],
      "metadata": {
        "id": "6XAN0h9dZDRX"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##ExampleUse"
      ],
      "metadata": {
        "id": "IHK4J3_MZomd"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Initial state\n",
        "initial_state = {\n",
        "    'user_id': 'analyst1',\n",
        "    'incident_description': (\n",
        "        'We have detected unusual activity from IP 203.0.113.42 accessing sensitive data. '\n",
        "        'Additionally, several phishing emails were sent from compromised accounts to external recipients. '\n",
        "        'Suspicious file hashes have been identified in our system.'\n",
        "    ),\n",
        "    'messages': [],\n",
        "    'max_analysts': 3\n",
        "}\n",
        "\n",
        "# Config\n",
        "config = {'configurable': {'thread_id': 'dlp_thread1'}}\n",
        "\n",
        "# Execute the graph up to the interruption point\n",
        "result_state = dlp_graph.invoke(initial_state, config)\n"
      ],
      "metadata": {
        "id": "Tc1v92jzZRPg"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"\\nGenerated Analysts for Review:\")\n",
        "for analyst in result_state['analysts']:\n",
        "    print(f\"Name: {analyst.name}\")\n",
        "    print(f\"Role: {analyst.role}\")\n",
        "    print(f\"Expertise: {analyst.expertise}\")\n",
        "    print(f\"Experience: {analyst.experience_years} years\")\n",
        "    print(f\"Certifications: {', '.join(analyst.certifications)}\")\n",
        "    print(f\"Personality Traits: {', '.join(analyst.personality_traits)}\")\n",
        "    print(f\"Description: {analyst.description}\")\n",
        "    print(\"-\" * 80)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xF0_opEkZt3Y",
        "outputId": "aabb5d89-4474-4677-b075-eb6ffe12f326"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Generated Analysts for Review:\n",
            "Name: Alex Mercer\n",
            "Role: Network Security Analyst\n",
            "Expertise: Network Intrusion Detection and Prevention\n",
            "Experience: 8 years\n",
            "Certifications: CISSP, CEH\n",
            "Personality Traits: Analytical, Detail-oriented, Persistent\n",
            "Description: Alex specializes in identifying and mitigating network-based threats. His focus in this incident will be on the unusual activity from IP 203.0.113.42. He will analyze network logs, identify the nature of the activity, and implement measures to prevent further intrusion. Alex's meticulous nature and persistence make him adept at uncovering hidden threats and vulnerabilities.\n",
            "--------------------------------------------------------------------------------\n",
            "Name: Samantha Reed\n",
            "Role: Cyber Threat Intelligence Analyst\n",
            "Expertise: Phishing and Email Security\n",
            "Experience: 6 years\n",
            "Certifications: GCIH, CEH\n",
            "Personality Traits: Innovative, Proactive, Adaptable\n",
            "Description: Samantha's expertise lies in identifying and mitigating email-based threats. In this incident, she will focus on the phishing emails sent from compromised accounts. She will work to identify the source of the phishing emails, secure the compromised accounts, and implement measures to prevent future phishing attempts. Samantha's innovative thinking and adaptability allow her to stay ahead of evolving cyber threats.\n",
            "--------------------------------------------------------------------------------\n",
            "Name: David Chen\n",
            "Role: Forensic Analyst\n",
            "Expertise: Malware Analysis and Digital Forensics\n",
            "Experience: 10 years\n",
            "Certifications: GCFA, CHFI\n",
            "Personality Traits: Cautious, Thorough, Patient\n",
            "Description: David specializes in malware analysis and digital forensics. His focus in this incident will be on the suspicious file hashes identified in the system. He will analyze these files, identify any potential malware, and determine its purpose and origin. David's cautious and thorough approach ensures that no detail is overlooked in the investigation.\n",
            "--------------------------------------------------------------------------------\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Provide human feedback (if any)\n",
        "dlp_graph.update_state(\n",
        "    config,\n",
        "    {'human_feedback': None},  # Set to None or provide feedback as a string\n",
        "    as_node='collect_human_feedback'\n",
        ")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2BGy7mRaZ2f1",
        "outputId": "eb4c7259-b5ba-4760-c0b1-692d982c225c"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'configurable': {'thread_id': 'dlp_thread1',\n",
              "  'checkpoint_ns': '',\n",
              "  'checkpoint_id': '1efaa90d-bff9-644c-8003-6ca37ff95c3a'}}"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Resume execution from the interruption point\n",
        "for _ in dlp_graph.stream(None, config, stream_mode='updates'):\n",
        "    pass\n"
      ],
      "metadata": {
        "id": "TyGyfPYQZ8_V"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Get the final state\n",
        "final_state = dlp_graph.get_state(config)[0]\n",
        "\n",
        "# Access and print the final report\n",
        "print(\"\\nDLP Incident Report:\")\n",
        "print(final_state['final_report'])\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Kc6IEsPIaADr",
        "outputId": "75c66159-0635-4036-c9cb-84e657cc405a"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "DLP Incident Report:\n",
            "# Incident Report\n",
            "\n",
            "## Analysis\n",
            "\n",
            "**Analyst:** Alex Mercer\n",
            "**Threat:** 203.0.113.42 (malicious_ip)\n",
            "Analysis Report\n",
            "\n",
            "Subject: Detailed Analysis of Threat - 203.0.113.42 (malicious_ip)\n",
            "\n",
            "1. Findings:\n",
            "\n",
            "After conducting a thorough analysis of the IP address 203.0.113.42, it has been identified as a source of malicious activity. The IP address is associated with numerous reported incidents of network intrusion attempts, distributed denial of service (DDoS) attacks, and phishing campaigns. \n",
            "\n",
            "The IP address is not associated with any known legitimate service provider, which further raises suspicion about its activities. The geolocation of the IP address is inconsistent, suggesting the use of advanced evasion techniques or a botnet infrastructure. \n",
            "\n",
            "The IP address has also been flagged in several threat intelligence databases for its association with malware distribution, specifically ransomware and trojans. The IP address has been seen to frequently change its attack patterns, indicating a high level of sophistication and adaptability.\n",
            "\n",
            "2. Potential Impact:\n",
            "\n",
            "The potential impact of this threat is significant. If successful, the intrusion attempts could lead to unauthorized access to sensitive data, disruption of services, and potential installation of malicious software. The DDoS attacks could lead to significant downtime, affecting business continuity. \n",
            "\n",
            "The phishing campaigns associated with this IP address could lead to credential theft, further exposing the network to additional threats. The distribution of ransomware and trojans could lead to data loss, financial loss due to ransom payments, and further compromise of the network.\n",
            "\n",
            "3. Severity Assessment:\n",
            "\n",
            "Given the findings and the potential impact, the severity of this threat is assessed as 'Critical'. The IP address is associated with a wide range of malicious activities, indicating a highly capable and persistent threat actor. The use of advanced evasion techniques and the adaptability of the attack patterns further increase the severity of this threat.\n",
            "\n",
            "4. Additional Observations:\n",
            "\n",
            "The level of sophistication and persistence associated with this IP address suggests that it may be part of a larger, coordinated cybercrime operation. The frequent change in attack patterns indicates a high level of adaptability, suggesting that the threat actor is continuously updating their tactics in response to defensive measures. \n",
            "\n",
            "The inconsistent geolocation could be indicative of the use of a botnet or a VPN to mask the true origin of the attacks. This level of obfuscation further increases the complexity of the threat.\n",
            "\n",
            "In conclusion, the threat posed by the IP address 203.0.113.42 is critical and requires immediate attention. The threat actor's persistence, adaptability, and use of advanced techniques indicate a high level of threat capability.\n",
            "\n",
            "**Analyst:** Samantha Reed\n",
            "**Threat:** compromised accounts (phishing_email)\n",
            "Subject: Threat Analysis Report - Compromised Accounts (Phishing_Email)\n",
            "\n",
            "1. Findings:\n",
            "\n",
            "Upon analysis of the phishing_email, it was found that the email was designed to mimic a legitimate communication from a trusted source. The email contained a link that redirected users to a fraudulent website, designed to capture the user's login credentials. The email was crafted with a sense of urgency, a common tactic used by threat actors to pressure the recipient into taking immediate action without verifying the authenticity of the email.\n",
            "\n",
            "The email headers were manipulated to appear as if they originated from a trusted source. The IP address associated with the email was traced back to a region known for hosting malicious servers. The fraudulent website was hosted on a recently registered domain, another common characteristic of phishing campaigns.\n",
            "\n",
            "2. Potential Impact:\n",
            "\n",
            "The potential impact of this phishing campaign is significant. If successful, the threat actors could gain unauthorized access to sensitive user accounts. This could lead to data breaches, identity theft, financial loss, and damage to the organization's reputation. The compromised accounts could also be used to launch further attacks, both within the organization and against its partners or customers.\n",
            "\n",
            "3. Severity Assessment:\n",
            "\n",
            "The severity of this threat is assessed as HIGH. The phishing email is well-crafted and could easily deceive users into providing their login credentials. The potential for significant data loss and financial impact, coupled with the potential for further attacks, makes this a serious threat that requires immediate attention.\n",
            "\n",
            "4. Additional Observations:\n",
            "\n",
            "The phishing email appears to be part of a larger campaign, as it uses tactics and infrastructure commonly associated with organized cybercrime groups. The use of a recently registered domain and the location of the IP address suggest that the threat actors are sophisticated and have invested resources into this attack.\n",
            "\n",
            "The email was not detected by standard spam filters, indicating that the threat actors have found ways to bypass these security measures. This suggests that the threat actors are continuously evolving their tactics to evade detection, highlighting the need for ongoing vigilance and proactive threat intelligence.\n",
            "\n",
            "In conclusion, the phishing_email represents a high-severity threat that could have significant impacts if not addressed promptly. The sophistication of the attack indicates that the threat actors are well-resourced and persistent, and are likely to continue their activities in the future.\n",
            "\n",
            "**Analyst:** David Chen\n",
            "**Threat:** suspicious file hashes (suspicious_file_hash)\n",
            "Subject: Detailed Analysis Report on Suspicious File Hashes\n",
            "\n",
            "Dear Team,\n",
            "\n",
            "I have completed a thorough analysis of the suspicious file hashes (suspicious_file_hash) that were recently flagged in our system. Here are my findings:\n",
            "\n",
            "1. Findings:\n",
            "\n",
            "Upon careful examination, I found that these file hashes are associated with known malicious software. The hashes match with those of a notorious Trojan horse malware, which is designed to provide unauthorized remote access to an attacker. The malware is known for its stealthy characteristics, making it difficult to detect.\n",
            "\n",
            "2. Potential Impact:\n",
            "\n",
            "The potential impact of this threat is high. If the malware is active, it could lead to a variety of negative outcomes. These include unauthorized access to sensitive data, loss of data integrity, and potential system downtime. The attacker could potentially gain control over the infected system, manipulate data, install additional malicious software, or even use the system as a launchpad for attacks on other systems within the network.\n",
            "\n",
            "3. Severity Assessment:\n",
            "\n",
            "Given the nature of the threat and its potential impact, I would classify the severity of this threat as 'High'. The stealthy nature of the Trojan horse malware, combined with its potential to cause significant damage, makes it a serious threat that requires immediate attention.\n",
            "\n",
            "4. Additional Observations:\n",
            "\n",
            "During my analysis, I noticed that the suspicious file hashes were found in multiple systems across our network. This suggests that the malware may have already spread within our infrastructure. It's also worth noting that this particular Trojan horse malware is known for its ability to evade detection by many common antivirus solutions, which may explain why it was not detected earlier.\n",
            "\n",
            "In conclusion, the suspicious file hashes represent a significant threat to our systems and data. The potential for unauthorized access and control by an attacker, coupled with the malware's stealthy characteristics, makes this a high-severity threat. \n",
            "\n",
            "As always, I approached this analysis with caution, thoroughness, and patience, ensuring that every possible angle was considered. I will continue to monitor the situation closely and provide updates as necessary.\n",
            "\n",
            "Best Regards,\n",
            "\n",
            "David Chen\n",
            "Forensic Analyst, Malware Analysis and Digital Forensics\n",
            "Certifications: GCFA, CHFI\n",
            "\n",
            "## Remediation Steps\n",
            "\n",
            "**Alex Mercer recommends:** Remediation Steps:\n",
            "\n",
            "1. Block IP Address: The first step is to block the IP address 203.0.113.42 at the firewall level. This will prevent any further communication between the malicious IP and your network. \n",
            "\n",
            "2. Update Intrusion Detection System (IDS) and Intrusion Prevention System (IPS): Update your IDS/IPS with the signature of the attack associated with the malicious IP. This will help in detecting and preventing similar attacks in the future.\n",
            "\n",
            "3. Patch Vulnerabilities: If the malicious IP was able to exploit a specific vulnerability in your system, ensure that the vulnerability is patched. Regularly update and patch all systems to prevent exploitation.\n",
            "\n",
            "4. Network Segmentation: Implement network segmentation to limit the spread of an attack within the network. This will ensure that even if one part of the network is compromised, the entire network will not be affected.\n",
            "\n",
            "5. Incident Response: Conduct a thorough investigation to understand the extent of the breach. This includes identifying compromised systems, the data that was accessed, and how the breach occurred. \n",
            "\n",
            "6. User Awareness: Train users on the importance of cybersecurity and how to identify potential threats. This will help in preventing future attacks.\n",
            "\n",
            "7. Regular Audits: Conduct regular audits of your network to identify any potential vulnerabilities or breaches. \n",
            "\n",
            "Potential Challenges:\n",
            "\n",
            "1. Time and Resources: Implementing these remediation steps requires time and resources. It may also require downtime, which could impact business operations.\n",
            "\n",
            "2. Technical Expertise: Some of these steps, such as updating the IDS/IPS and patching vulnerabilities, require technical expertise. If you do not have the necessary expertise in-house, you may need to hire an external consultant.\n",
            "\n",
            "3. User Resistance: Users may resist changes, especially if they perceive them as making their work more difficult. It's important to communicate the importance of these changes and provide training to help users adapt.\n",
            "\n",
            "4. Ongoing Maintenance: Cybersecurity is not a one-time task. It requires ongoing maintenance and vigilance to ensure that your network remains secure. This includes regularly updating and patching systems, monitoring for suspicious activity, and conducting regular audits.\n",
            "**Samantha Reed recommends:** Based on the analysis of the compromised accounts, the following remediation steps are recommended:\n",
            "\n",
            "1. **Password Reset**: The first step is to reset the passwords for all compromised accounts. This should be done immediately to prevent further unauthorized access. The new passwords should be strong, with a mix of uppercase and lowercase letters, numbers, and special characters.\n",
            "\n",
            "2. **Two-Factor Authentication (2FA)**: Implement two-factor authentication for all accounts. This adds an extra layer of security by requiring users to verify their identity with a second factor, such as a text message or an app on their phone, in addition to their password.\n",
            "\n",
            "3. **Email Security Training**: Conduct training sessions for all employees to educate them about phishing attacks and how to identify suspicious emails. This should include information about the common signs of phishing emails, such as poor grammar and spelling, requests for personal information, and suspicious links or attachments.\n",
            "\n",
            "4. **Email Filtering**: Implement an email filtering solution that can detect and block phishing emails before they reach the user's inbox. This can help to reduce the risk of future attacks.\n",
            "\n",
            "5. **Incident Response Plan**: Develop an incident response plan that outlines the steps to take in the event of a future phishing attack. This should include procedures for identifying and isolating compromised accounts, notifying affected users, and reporting the incident to the appropriate authorities.\n",
            "\n",
            "6. **Regular Audits**: Conduct regular audits of your email system to identify any unusual activity that could indicate a phishing attack. This can help to detect attacks early and minimize their impact.\n",
            "\n",
            "Potential challenges in implementing these remediation steps include:\n",
            "\n",
            "- **User Resistance**: Users may resist changes such as password resets and two-factor authentication, particularly if they find them inconvenient. This can be addressed through education about the importance of these measures for their security.\n",
            "\n",
            "- **Resource Constraints**: Implementing new security measures and conducting regular audits can be resource-intensive. It may be necessary to allocate additional resources or seek external assistance.\n",
            "\n",
            "- **Technical Challenges**: Depending on the complexity of your email system, there may be technical challenges in implementing measures such as email filtering and regular audits. These can be addressed by seeking advice from IT professionals or cybersecurity consultants. \n",
            "\n",
            "- **Ongoing Vigilance**: Phishing tactics are constantly evolving, so it's important to stay up-to-date with the latest threats and adjust your security measures accordingly. This requires ongoing vigilance and a commitment to continuous learning and improvement.\n",
            "**David Chen recommends:** Based on the analysis of the suspicious file hashes, the following remediation steps are recommended:\n",
            "\n",
            "1. **Isolation**: The first step is to isolate the affected systems to prevent the potential spread of malware. This can be done by disconnecting the system from the network. \n",
            "\n",
            "2. **Backup**: Make a backup of the system for further analysis and to preserve any potential evidence. This should be done before any remediation steps are taken.\n",
            "\n",
            "3. **Identification**: Use the suspicious file hashes to identify the malware. This can be done using various online databases that contain information about known malware, such as VirusTotal or any other threat intelligence platform.\n",
            "\n",
            "4. **Removal**: Use a reputable antivirus or anti-malware tool to remove the identified malware. This should be done in safe mode to prevent the malware from interfering with the removal process.\n",
            "\n",
            "5. **System Update**: Update the system and all installed software to the latest versions. This can help to patch any vulnerabilities that the malware may have exploited.\n",
            "\n",
            "6. **Password Change**: Change all passwords on the system. This is because malware often tries to steal passwords.\n",
            "\n",
            "7. **Network Monitoring**: Monitor network traffic for any signs of unusual activity. This can help to identify if the malware has spread to other systems.\n",
            "\n",
            "8. **Education**: Educate users about the dangers of malware and how to avoid it. This includes not opening suspicious emails, not downloading files from untrusted sources, and keeping software up to date.\n",
            "\n",
            "Potential challenges in implementing these remediation steps include:\n",
            "\n",
            "- **User Resistance**: Users may resist changes, especially if they involve changing passwords or learning new procedures. This can be mitigated by explaining the reasons for the changes and providing training if necessary.\n",
            "\n",
            "- **System Downtime**: The remediation process may require system downtime, which can disrupt business operations. This can be mitigated by scheduling the remediation during off-peak hours.\n",
            "\n",
            "- **Incomplete Removal**: Some malware is designed to resist removal, and may re-infect the system even after the remediation steps have been taken. This can be mitigated by using multiple antivirus tools and by monitoring the system for signs of re-infection.\n",
            "\n",
            "- **Data Loss**: There is a risk of data loss during the remediation process, especially if the malware has encrypted or deleted files. This can be mitigated by making a backup before starting the remediation process.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from IPython.display import display, Markdown\n",
        "\n",
        "# After generating the final report\n",
        "final_state = dlp_graph.get_state(config)[0]\n",
        "\n",
        "# Pretty print the final report\n",
        "display(Markdown(final_state['final_report']))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "FpIdCHqOaDFn",
        "outputId": "e498e0f9-41e0-43b7-bfd4-67cde3d2ef5f"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Markdown object>"
            ],
            "text/markdown": "# Incident Report\n\n## Analysis\n\n**Analyst:** Alex Mercer\n**Threat:** 203.0.113.42 (malicious_ip)\nAnalysis Report\n\nSubject: Detailed Analysis of Threat - 203.0.113.42 (malicious_ip)\n\n1. Findings:\n\nAfter conducting a thorough analysis of the IP address 203.0.113.42, it has been identified as a source of malicious activity. The IP address is associated with numerous reported incidents of network intrusion attempts, distributed denial of service (DDoS) attacks, and phishing campaigns. \n\nThe IP address is not associated with any known legitimate service provider, which further raises suspicion about its activities. The geolocation of the IP address is inconsistent, suggesting the use of advanced evasion techniques or a botnet infrastructure. \n\nThe IP address has also been flagged in several threat intelligence databases for its association with malware distribution, specifically ransomware and trojans. The IP address has been seen to frequently change its attack patterns, indicating a high level of sophistication and adaptability.\n\n2. Potential Impact:\n\nThe potential impact of this threat is significant. If successful, the intrusion attempts could lead to unauthorized access to sensitive data, disruption of services, and potential installation of malicious software. The DDoS attacks could lead to significant downtime, affecting business continuity. \n\nThe phishing campaigns associated with this IP address could lead to credential theft, further exposing the network to additional threats. The distribution of ransomware and trojans could lead to data loss, financial loss due to ransom payments, and further compromise of the network.\n\n3. Severity Assessment:\n\nGiven the findings and the potential impact, the severity of this threat is assessed as 'Critical'. The IP address is associated with a wide range of malicious activities, indicating a highly capable and persistent threat actor. The use of advanced evasion techniques and the adaptability of the attack patterns further increase the severity of this threat.\n\n4. Additional Observations:\n\nThe level of sophistication and persistence associated with this IP address suggests that it may be part of a larger, coordinated cybercrime operation. The frequent change in attack patterns indicates a high level of adaptability, suggesting that the threat actor is continuously updating their tactics in response to defensive measures. \n\nThe inconsistent geolocation could be indicative of the use of a botnet or a VPN to mask the true origin of the attacks. This level of obfuscation further increases the complexity of the threat.\n\nIn conclusion, the threat posed by the IP address 203.0.113.42 is critical and requires immediate attention. The threat actor's persistence, adaptability, and use of advanced techniques indicate a high level of threat capability.\n\n**Analyst:** Samantha Reed\n**Threat:** compromised accounts (phishing_email)\nSubject: Threat Analysis Report - Compromised Accounts (Phishing_Email)\n\n1. Findings:\n\nUpon analysis of the phishing_email, it was found that the email was designed to mimic a legitimate communication from a trusted source. The email contained a link that redirected users to a fraudulent website, designed to capture the user's login credentials. The email was crafted with a sense of urgency, a common tactic used by threat actors to pressure the recipient into taking immediate action without verifying the authenticity of the email.\n\nThe email headers were manipulated to appear as if they originated from a trusted source. The IP address associated with the email was traced back to a region known for hosting malicious servers. The fraudulent website was hosted on a recently registered domain, another common characteristic of phishing campaigns.\n\n2. Potential Impact:\n\nThe potential impact of this phishing campaign is significant. If successful, the threat actors could gain unauthorized access to sensitive user accounts. This could lead to data breaches, identity theft, financial loss, and damage to the organization's reputation. The compromised accounts could also be used to launch further attacks, both within the organization and against its partners or customers.\n\n3. Severity Assessment:\n\nThe severity of this threat is assessed as HIGH. The phishing email is well-crafted and could easily deceive users into providing their login credentials. The potential for significant data loss and financial impact, coupled with the potential for further attacks, makes this a serious threat that requires immediate attention.\n\n4. Additional Observations:\n\nThe phishing email appears to be part of a larger campaign, as it uses tactics and infrastructure commonly associated with organized cybercrime groups. The use of a recently registered domain and the location of the IP address suggest that the threat actors are sophisticated and have invested resources into this attack.\n\nThe email was not detected by standard spam filters, indicating that the threat actors have found ways to bypass these security measures. This suggests that the threat actors are continuously evolving their tactics to evade detection, highlighting the need for ongoing vigilance and proactive threat intelligence.\n\nIn conclusion, the phishing_email represents a high-severity threat that could have significant impacts if not addressed promptly. The sophistication of the attack indicates that the threat actors are well-resourced and persistent, and are likely to continue their activities in the future.\n\n**Analyst:** David Chen\n**Threat:** suspicious file hashes (suspicious_file_hash)\nSubject: Detailed Analysis Report on Suspicious File Hashes\n\nDear Team,\n\nI have completed a thorough analysis of the suspicious file hashes (suspicious_file_hash) that were recently flagged in our system. Here are my findings:\n\n1. Findings:\n\nUpon careful examination, I found that these file hashes are associated with known malicious software. The hashes match with those of a notorious Trojan horse malware, which is designed to provide unauthorized remote access to an attacker. The malware is known for its stealthy characteristics, making it difficult to detect.\n\n2. Potential Impact:\n\nThe potential impact of this threat is high. If the malware is active, it could lead to a variety of negative outcomes. These include unauthorized access to sensitive data, loss of data integrity, and potential system downtime. The attacker could potentially gain control over the infected system, manipulate data, install additional malicious software, or even use the system as a launchpad for attacks on other systems within the network.\n\n3. Severity Assessment:\n\nGiven the nature of the threat and its potential impact, I would classify the severity of this threat as 'High'. The stealthy nature of the Trojan horse malware, combined with its potential to cause significant damage, makes it a serious threat that requires immediate attention.\n\n4. Additional Observations:\n\nDuring my analysis, I noticed that the suspicious file hashes were found in multiple systems across our network. This suggests that the malware may have already spread within our infrastructure. It's also worth noting that this particular Trojan horse malware is known for its ability to evade detection by many common antivirus solutions, which may explain why it was not detected earlier.\n\nIn conclusion, the suspicious file hashes represent a significant threat to our systems and data. The potential for unauthorized access and control by an attacker, coupled with the malware's stealthy characteristics, makes this a high-severity threat. \n\nAs always, I approached this analysis with caution, thoroughness, and patience, ensuring that every possible angle was considered. I will continue to monitor the situation closely and provide updates as necessary.\n\nBest Regards,\n\nDavid Chen\nForensic Analyst, Malware Analysis and Digital Forensics\nCertifications: GCFA, CHFI\n\n## Remediation Steps\n\n**Alex Mercer recommends:** Remediation Steps:\n\n1. Block IP Address: The first step is to block the IP address 203.0.113.42 at the firewall level. This will prevent any further communication between the malicious IP and your network. \n\n2. Update Intrusion Detection System (IDS) and Intrusion Prevention System (IPS): Update your IDS/IPS with the signature of the attack associated with the malicious IP. This will help in detecting and preventing similar attacks in the future.\n\n3. Patch Vulnerabilities: If the malicious IP was able to exploit a specific vulnerability in your system, ensure that the vulnerability is patched. Regularly update and patch all systems to prevent exploitation.\n\n4. Network Segmentation: Implement network segmentation to limit the spread of an attack within the network. This will ensure that even if one part of the network is compromised, the entire network will not be affected.\n\n5. Incident Response: Conduct a thorough investigation to understand the extent of the breach. This includes identifying compromised systems, the data that was accessed, and how the breach occurred. \n\n6. User Awareness: Train users on the importance of cybersecurity and how to identify potential threats. This will help in preventing future attacks.\n\n7. Regular Audits: Conduct regular audits of your network to identify any potential vulnerabilities or breaches. \n\nPotential Challenges:\n\n1. Time and Resources: Implementing these remediation steps requires time and resources. It may also require downtime, which could impact business operations.\n\n2. Technical Expertise: Some of these steps, such as updating the IDS/IPS and patching vulnerabilities, require technical expertise. If you do not have the necessary expertise in-house, you may need to hire an external consultant.\n\n3. User Resistance: Users may resist changes, especially if they perceive them as making their work more difficult. It's important to communicate the importance of these changes and provide training to help users adapt.\n\n4. Ongoing Maintenance: Cybersecurity is not a one-time task. It requires ongoing maintenance and vigilance to ensure that your network remains secure. This includes regularly updating and patching systems, monitoring for suspicious activity, and conducting regular audits.\n**Samantha Reed recommends:** Based on the analysis of the compromised accounts, the following remediation steps are recommended:\n\n1. **Password Reset**: The first step is to reset the passwords for all compromised accounts. This should be done immediately to prevent further unauthorized access. The new passwords should be strong, with a mix of uppercase and lowercase letters, numbers, and special characters.\n\n2. **Two-Factor Authentication (2FA)**: Implement two-factor authentication for all accounts. This adds an extra layer of security by requiring users to verify their identity with a second factor, such as a text message or an app on their phone, in addition to their password.\n\n3. **Email Security Training**: Conduct training sessions for all employees to educate them about phishing attacks and how to identify suspicious emails. This should include information about the common signs of phishing emails, such as poor grammar and spelling, requests for personal information, and suspicious links or attachments.\n\n4. **Email Filtering**: Implement an email filtering solution that can detect and block phishing emails before they reach the user's inbox. This can help to reduce the risk of future attacks.\n\n5. **Incident Response Plan**: Develop an incident response plan that outlines the steps to take in the event of a future phishing attack. This should include procedures for identifying and isolating compromised accounts, notifying affected users, and reporting the incident to the appropriate authorities.\n\n6. **Regular Audits**: Conduct regular audits of your email system to identify any unusual activity that could indicate a phishing attack. This can help to detect attacks early and minimize their impact.\n\nPotential challenges in implementing these remediation steps include:\n\n- **User Resistance**: Users may resist changes such as password resets and two-factor authentication, particularly if they find them inconvenient. This can be addressed through education about the importance of these measures for their security.\n\n- **Resource Constraints**: Implementing new security measures and conducting regular audits can be resource-intensive. It may be necessary to allocate additional resources or seek external assistance.\n\n- **Technical Challenges**: Depending on the complexity of your email system, there may be technical challenges in implementing measures such as email filtering and regular audits. These can be addressed by seeking advice from IT professionals or cybersecurity consultants. \n\n- **Ongoing Vigilance**: Phishing tactics are constantly evolving, so it's important to stay up-to-date with the latest threats and adjust your security measures accordingly. This requires ongoing vigilance and a commitment to continuous learning and improvement.\n**David Chen recommends:** Based on the analysis of the suspicious file hashes, the following remediation steps are recommended:\n\n1. **Isolation**: The first step is to isolate the affected systems to prevent the potential spread of malware. This can be done by disconnecting the system from the network. \n\n2. **Backup**: Make a backup of the system for further analysis and to preserve any potential evidence. This should be done before any remediation steps are taken.\n\n3. **Identification**: Use the suspicious file hashes to identify the malware. This can be done using various online databases that contain information about known malware, such as VirusTotal or any other threat intelligence platform.\n\n4. **Removal**: Use a reputable antivirus or anti-malware tool to remove the identified malware. This should be done in safe mode to prevent the malware from interfering with the removal process.\n\n5. **System Update**: Update the system and all installed software to the latest versions. This can help to patch any vulnerabilities that the malware may have exploited.\n\n6. **Password Change**: Change all passwords on the system. This is because malware often tries to steal passwords.\n\n7. **Network Monitoring**: Monitor network traffic for any signs of unusual activity. This can help to identify if the malware has spread to other systems.\n\n8. **Education**: Educate users about the dangers of malware and how to avoid it. This includes not opening suspicious emails, not downloading files from untrusted sources, and keeping software up to date.\n\nPotential challenges in implementing these remediation steps include:\n\n- **User Resistance**: Users may resist changes, especially if they involve changing passwords or learning new procedures. This can be mitigated by explaining the reasons for the changes and providing training if necessary.\n\n- **System Downtime**: The remediation process may require system downtime, which can disrupt business operations. This can be mitigated by scheduling the remediation during off-peak hours.\n\n- **Incomplete Removal**: Some malware is designed to resist removal, and may re-infect the system even after the remediation steps have been taken. This can be mitigated by using multiple antivirus tools and by monitoring the system for signs of re-infection.\n\n- **Data Loss**: There is a risk of data loss during the remediation process, especially if the malware has encrypted or deleted files. This can be mitigated by making a backup before starting the remediation process."
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install gradio\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jB7l9qM1cTn1",
        "outputId": "a29cd1d0-4fd7-46ae-b0b4-47f43c94bd5d"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting gradio\n",
            "  Downloading gradio-5.6.0-py3-none-any.whl.metadata (16 kB)\n",
            "Collecting aiofiles<24.0,>=22.0 (from gradio)\n",
            "  Downloading aiofiles-23.2.1-py3-none-any.whl.metadata (9.7 kB)\n",
            "Requirement already satisfied: anyio<5.0,>=3.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.7.1)\n",
            "Collecting fastapi<1.0,>=0.115.2 (from gradio)\n",
            "  Downloading fastapi-0.115.5-py3-none-any.whl.metadata (27 kB)\n",
            "Collecting ffmpy (from gradio)\n",
            "  Downloading ffmpy-0.4.0-py3-none-any.whl.metadata (2.9 kB)\n",
            "Collecting gradio-client==1.4.3 (from gradio)\n",
            "  Downloading gradio_client-1.4.3-py3-none-any.whl.metadata (7.1 kB)\n",
            "Requirement already satisfied: httpx>=0.24.1 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.27.2)\n",
            "Requirement already satisfied: huggingface-hub>=0.25.1 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.26.2)\n",
            "Requirement already satisfied: jinja2<4.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.1.4)\n",
            "Collecting markupsafe~=2.0 (from gradio)\n",
            "  Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)\n",
            "Requirement already satisfied: numpy<3.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (1.26.4)\n",
            "Requirement already satisfied: orjson~=3.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (3.10.11)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from gradio) (24.2)\n",
            "Requirement already satisfied: pandas<3.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.2.2)\n",
            "Requirement already satisfied: pillow<12.0,>=8.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (11.0.0)\n",
            "Requirement already satisfied: pydantic>=2.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (2.9.2)\n",
            "Collecting pydub (from gradio)\n",
            "  Downloading pydub-0.25.1-py2.py3-none-any.whl.metadata (1.4 kB)\n",
            "Collecting python-multipart==0.0.12 (from gradio)\n",
            "  Downloading python_multipart-0.0.12-py3-none-any.whl.metadata (1.9 kB)\n",
            "Requirement already satisfied: pyyaml<7.0,>=5.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (6.0.2)\n",
            "Collecting ruff>=0.2.2 (from gradio)\n",
            "  Downloading ruff-0.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (25 kB)\n",
            "Collecting safehttpx<1.0,>=0.1.1 (from gradio)\n",
            "  Downloading safehttpx-0.1.1-py3-none-any.whl.metadata (4.1 kB)\n",
            "Collecting semantic-version~=2.0 (from gradio)\n",
            "  Downloading semantic_version-2.10.0-py2.py3-none-any.whl.metadata (9.7 kB)\n",
            "Collecting starlette<1.0,>=0.40.0 (from gradio)\n",
            "  Downloading starlette-0.41.3-py3-none-any.whl.metadata (6.0 kB)\n",
            "Collecting tomlkit==0.12.0 (from gradio)\n",
            "  Downloading tomlkit-0.12.0-py3-none-any.whl.metadata (2.7 kB)\n",
            "Requirement already satisfied: typer<1.0,>=0.12 in /usr/local/lib/python3.10/dist-packages (from gradio) (0.13.0)\n",
            "Requirement already satisfied: typing-extensions~=4.0 in /usr/local/lib/python3.10/dist-packages (from gradio) (4.12.2)\n",
            "Collecting uvicorn>=0.14.0 (from gradio)\n",
            "  Downloading uvicorn-0.32.1-py3-none-any.whl.metadata (6.6 kB)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from gradio-client==1.4.3->gradio) (2024.10.0)\n",
            "Collecting websockets<13.0,>=10.0 (from gradio-client==1.4.3->gradio)\n",
            "  Downloading websockets-12.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)\n",
            "Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (3.10)\n",
            "Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)\n",
            "Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5.0,>=3.0->gradio) (1.2.2)\n",
            "Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx>=0.24.1->gradio) (2024.8.30)\n",
            "Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx>=0.24.1->gradio) (1.0.7)\n",
            "Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.14.0)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.25.1->gradio) (3.16.1)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.25.1->gradio) (2.32.3)\n",
            "Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub>=0.25.1->gradio) (4.66.6)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas<3.0,>=1.0->gradio) (2024.2)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic>=2.0->gradio) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic>=2.0->gradio) (2.23.4)\n",
            "Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (8.1.7)\n",
            "Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (1.5.4)\n",
            "Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from typer<1.0,>=0.12->gradio) (13.9.4)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3.0,>=1.0->gradio) (1.16.0)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.18.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.25.1->gradio) (3.4.0)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub>=0.25.1->gradio) (2.2.3)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.2)\n",
            "Downloading gradio-5.6.0-py3-none-any.whl (57.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m57.1/57.1 MB\u001b[0m \u001b[31m12.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading gradio_client-1.4.3-py3-none-any.whl (320 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m320.1/320.1 kB\u001b[0m \u001b[31m22.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading python_multipart-0.0.12-py3-none-any.whl (23 kB)\n",
            "Downloading tomlkit-0.12.0-py3-none-any.whl (37 kB)\n",
            "Downloading aiofiles-23.2.1-py3-none-any.whl (15 kB)\n",
            "Downloading fastapi-0.115.5-py3-none-any.whl (94 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m94.9/94.9 kB\u001b[0m \u001b[31m6.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)\n",
            "Downloading ruff-0.8.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m11.1/11.1 MB\u001b[0m \u001b[31m103.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading safehttpx-0.1.1-py3-none-any.whl (8.4 kB)\n",
            "Downloading semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)\n",
            "Downloading starlette-0.41.3-py3-none-any.whl (73 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m73.2/73.2 kB\u001b[0m \u001b[31m5.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading uvicorn-0.32.1-py3-none-any.whl (63 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m63.8/63.8 kB\u001b[0m \u001b[31m4.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading ffmpy-0.4.0-py3-none-any.whl (5.8 kB)\n",
            "Downloading pydub-0.25.1-py2.py3-none-any.whl (32 kB)\n",
            "Downloading websockets-12.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m130.2/130.2 kB\u001b[0m \u001b[31m10.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydub, websockets, uvicorn, tomlkit, semantic-version, ruff, python-multipart, markupsafe, ffmpy, aiofiles, starlette, safehttpx, gradio-client, fastapi, gradio\n",
            "  Attempting uninstall: markupsafe\n",
            "    Found existing installation: MarkupSafe 3.0.2\n",
            "    Uninstalling MarkupSafe-3.0.2:\n",
            "      Successfully uninstalled MarkupSafe-3.0.2\n",
            "Successfully installed aiofiles-23.2.1 fastapi-0.115.5 ffmpy-0.4.0 gradio-5.6.0 gradio-client-1.4.3 markupsafe-2.1.5 pydub-0.25.1 python-multipart-0.0.12 ruff-0.8.0 safehttpx-0.1.1 semantic-version-2.10.0 starlette-0.41.3 tomlkit-0.12.0 uvicorn-0.32.1 websockets-12.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Example of run_dlp_app function adjusted for Gradio\n",
        "def run_dlp_app(incident_description, max_analysts):\n",
        "    # Initialize state\n",
        "    initial_state = {\n",
        "        'user_id': 'analyst1',\n",
        "        'incident_description': incident_description,\n",
        "        'messages': [],\n",
        "        'max_analysts': int(max_analysts)\n",
        "    }\n",
        "\n",
        "    # Config\n",
        "    config = {'configurable': {'thread_id': 'dlp_thread1'}}\n",
        "\n",
        "    # Execute the graph up to the interruption point\n",
        "    result_state = dlp_graph.invoke(initial_state, config)\n",
        "\n",
        "    # For simplicity, we'll skip human feedback\n",
        "    dlp_graph.update_state(\n",
        "        config,\n",
        "        {'human_feedback': None},\n",
        "        as_node='collect_human_feedback'\n",
        "    )\n",
        "\n",
        "    # Resume execution from the interruption point\n",
        "    for _ in dlp_graph.stream(None, config, stream_mode='updates'):\n",
        "        pass\n",
        "\n",
        "    # Get the final state\n",
        "    final_state = dlp_graph.get_state(config)[0]\n",
        "\n",
        "    # Return the final report\n",
        "    return final_state['final_report']\n",
        "\n",
        "iface = gr.Interface(\n",
        "    fn=run_dlp_app,\n",
        "    inputs=[\n",
        "        gr.Textbox(lines=10, label=\"Incident Description\"),\n",
        "        gr.Slider(minimum=1, maximum=5, value=3, step=1, label=\"Number of Analysts\")\n",
        "    ],\n",
        "    outputs=\"markdown\",\n",
        "    title=\"DLP Incident Analysis\",\n",
        "    description=\"Analyze cybersecurity incidents using AI-generated security analyst personas.\"\n",
        ")\n",
        "\n",
        "iface.launch()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 628
        },
        "id": "AJ1L3SJcc5g7",
        "outputId": "16cd3f29-cab5-4150-d85f-d5e8b39503f9"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Running Gradio in a Colab notebook requires sharing enabled. Automatically setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).\n",
            "\n",
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "* Running on public URL: https://0e1bab1a9e699349b8.gradio.live\n",
            "\n",
            "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://0e1bab1a9e699349b8.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Import necessary modules\n",
        "import os\n",
        "import getpass\n",
        "import gradio as gr\n",
        "import logging\n",
        "\n",
        "# Set up logging\n",
        "logging.basicConfig(level=logging.DEBUG)\n",
        "\n",
        "def _set_env(var: str, value: str = None):\n",
        "    if value:\n",
        "        os.environ[var] = value\n",
        "    elif not os.environ.get(var):\n",
        "        os.environ[var] = getpass.getpass(f\"Enter your {var}: \")\n",
        "\n",
        "# Set API keys\n",
        "_set_env(\"OPENAI_API_KEY\")\n",
        "_set_env(\"LANGCHAIN_API_KEY\")\n",
        "\n",
        "# Set LangChain tracing (optional)\n",
        "os.environ[\"LANGCHAIN_TRACING_V2\"] = \"true\"\n",
        "os.environ[\"LANGCHAIN_PROJECT\"] = \"dlp-agent\"\n",
        "\n",
        "from langchain_openai import ChatOpenAI\n",
        "llm = ChatOpenAI(model=\"gpt-4\", temperature=0)\n",
        "\n",
        "# Define data models\n",
        "from pydantic import BaseModel, Field\n",
        "from typing import List, Optional\n",
        "\n",
        "class Threat(BaseModel):\n",
        "    indicator: str = Field(description=\"Indicator of compromise (IP, domain, etc.)\")\n",
        "    type: str = Field(description=\"Type of threat\")\n",
        "    description: str = Field(default=\"\", description=\"Description of the threat\")\n",
        "    severity: str = Field(default=\"\", description=\"Severity level\")\n",
        "    status: str = Field(default=\"Pending\", description=\"Analysis status\")\n",
        "\n",
        "class SecurityAnalyst(BaseModel):\n",
        "    name: str = Field(description=\"Name of the analyst\")\n",
        "    role: str = Field(description=\"Role or job title\")\n",
        "    expertise: str = Field(description=\"Area of expertise\")\n",
        "    experience_years: int = Field(description=\"Years of experience\")\n",
        "    certifications: List[str] = Field(description=\"Certifications\")\n",
        "    personality_traits: List[str] = Field(description=\"Personality traits\")\n",
        "    description: str = Field(description=\"Detailed description of the analyst\")\n",
        "\n",
        "class ThreatList(BaseModel):\n",
        "    threats: List[Threat] = Field(description=\"List of extracted threats.\")\n",
        "\n",
        "class AnalystTeam(BaseModel):\n",
        "    analysts: List[SecurityAnalyst] = Field(description=\"List of security analysts.\")\n",
        "\n",
        "# Define state schema and helper functions\n",
        "from typing import TypedDict, Annotated\n",
        "from langchain_core.messages import AnyMessage\n",
        "from langgraph.graph.message import add_messages\n",
        "\n",
        "class DLPState(TypedDict):\n",
        "    user_id: str\n",
        "    messages: Annotated[List[AnyMessage], add_messages]\n",
        "    incident_description: str\n",
        "    threats: List[Threat]\n",
        "    human_feedback: Optional[str]\n",
        "    analysts: List[SecurityAnalyst]\n",
        "    analysis_sections: List[str]\n",
        "    remediation_steps: List[str]\n",
        "    final_report: str\n",
        "    max_analysts: int\n",
        "\n",
        "# Define prompt templates\n",
        "threat_extraction_instructions = \"\"\"\n",
        "You are a cybersecurity assistant. Your task is to identify potential threats from the incident description provided.\n",
        "\n",
        "Instructions:\n",
        "1. Extract all possible indicators of compromise (IOCs) such as IP addresses, domains, file hashes, email addresses, etc.\n",
        "2. For each IOC, determine its type (e.g., malicious_ip, phishing_email, suspicious_file_hash).\n",
        "3. Present the threats in a structured JSON format as a list of threats.\n",
        "\n",
        "Incident Description:\n",
        "{incident_description}\n",
        "\"\"\"\n",
        "\n",
        "analyst_generation_instructions = \"\"\"\n",
        "You are tasked with creating a team of AI security analyst personas for incident analysis.\n",
        "\n",
        "Instructions:\n",
        "1. Based on the incident description, create up to {max_analysts} analysts, each with different expertise relevant to the incident.\n",
        "2. For each analyst, provide:\n",
        "   - Name\n",
        "   - Role\n",
        "   - Expertise\n",
        "   - Years of experience\n",
        "   - Certifications\n",
        "   - Personality traits\n",
        "   - Description\n",
        "\n",
        "Present the analysts in a structured JSON format as a list.\n",
        "\n",
        "Incident Description:\n",
        "{incident_description}\n",
        "\"\"\"\n",
        "\n",
        "threat_analysis_instructions_template = \"\"\"\n",
        "You are {analyst_name}, a {role} specializing in {expertise} with {experience_years} years of experience.\n",
        "\n",
        "Instructions:\n",
        "1. Analyze the following threat: {threat_indicator} ({threat_type}).\n",
        "2. Provide a detailed analysis report including findings, potential impact, severity assessment, and additional observations.\n",
        "3. Use your personality traits: {personality_traits}.\n",
        "\n",
        "Do not include remediation steps.\n",
        "\"\"\"\n",
        "\n",
        "remediation_instructions_template = \"\"\"\n",
        "You are {analyst_name}, a {role} specializing in {expertise}.\n",
        "\n",
        "Instructions:\n",
        "1. Based on your analysis of {threat_indicator} ({threat_type}), provide recommended remediation steps.\n",
        "2. Be specific and consider best practices.\n",
        "3. Address potential challenges in implementing the remediation.\n",
        "\"\"\"\n",
        "\n",
        "# Define functions and nodes\n",
        "from langchain_core.messages import SystemMessage\n",
        "\n",
        "def extract_threats(state: DLPState):\n",
        "    incident_description = state['incident_description']\n",
        "    logging.debug(\"Starting threat extraction...\")\n",
        "    # Enforce structured output\n",
        "    structured_llm = llm.with_structured_output(ThreatList)\n",
        "    # System message\n",
        "    system_message = SystemMessage(content=threat_extraction_instructions.format(incident_description=incident_description))\n",
        "    # Invoke LLM\n",
        "    try:\n",
        "        response = structured_llm.invoke([system_message])\n",
        "        extracted_threats = response.threats\n",
        "        logging.debug(f\"Extracted threats: {extracted_threats}\")\n",
        "        return {'threats': extracted_threats}\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error during threat extraction: {e}\")\n",
        "        return {'threats': []}\n",
        "\n",
        "def create_analysts(state: DLPState):\n",
        "    incident_description = state['incident_description']\n",
        "    max_analysts = state.get('max_analysts', 3)\n",
        "    logging.debug(\"Starting analyst generation...\")\n",
        "    # Enforce structured output\n",
        "    structured_llm = llm.with_structured_output(AnalystTeam)\n",
        "    # System message\n",
        "    system_message_content = analyst_generation_instructions.format(\n",
        "        incident_description=incident_description,\n",
        "        max_analysts=max_analysts\n",
        "    )\n",
        "    system_message = SystemMessage(content=system_message_content)\n",
        "    # Invoke LLM\n",
        "    try:\n",
        "        response = structured_llm.invoke([system_message])\n",
        "        generated_analysts = response.analysts\n",
        "        logging.debug(f\"Generated analysts: {generated_analysts}\")\n",
        "        return {'analysts': generated_analysts}\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error during analyst generation: {e}\")\n",
        "        return {'analysts': []}\n",
        "\n",
        "def analyze_threats(state: DLPState):\n",
        "    threats = state['threats']\n",
        "    analysts = state['analysts']\n",
        "    analysis_sections = []\n",
        "    num_analysts = len(analysts)\n",
        "    logging.debug(\"Starting threat analysis...\")\n",
        "    if not threats or not analysts:\n",
        "        logging.error(\"No threats or analysts available for analysis.\")\n",
        "        return {'analysis_sections': []}\n",
        "    for idx, threat in enumerate(threats):\n",
        "        analyst = analysts[idx % num_analysts]\n",
        "        prompt = threat_analysis_instructions_template.format(\n",
        "            analyst_name=analyst.name,\n",
        "            role=analyst.role,\n",
        "            expertise=analyst.expertise,\n",
        "            experience_years=analyst.experience_years,\n",
        "            personality_traits=', '.join(analyst.personality_traits),\n",
        "            threat_indicator=threat.indicator,\n",
        "            threat_type=threat.type\n",
        "        )\n",
        "        system_message = SystemMessage(content=prompt)\n",
        "        try:\n",
        "            response = llm.invoke([system_message])\n",
        "            analysis = response.content.strip()\n",
        "            threat.description = analysis\n",
        "            threat.status = \"Analyzed\"\n",
        "            analysis_sections.append(\n",
        "                f\"**Analyst:** {analyst.name}\\n\"\n",
        "                f\"**Threat:** {threat.indicator} ({threat.type})\\n\"\n",
        "                f\"{analysis}\\n\"\n",
        "            )\n",
        "        except Exception as e:\n",
        "            logging.error(f\"Error during threat analysis for {threat.indicator}: {e}\")\n",
        "    logging.debug(\"Threat analysis completed.\")\n",
        "    return {'analysis_sections': analysis_sections}\n",
        "\n",
        "def suggest_remediation_steps(state: DLPState):\n",
        "    threats = state['threats']\n",
        "    analysts = state['analysts']\n",
        "    remediation_steps = []\n",
        "    num_analysts = len(analysts)\n",
        "    logging.debug(\"Starting remediation steps generation...\")\n",
        "    if not threats or not analysts:\n",
        "        logging.error(\"No threats or analysts available for remediation.\")\n",
        "        return {'remediation_steps': []}\n",
        "    for idx, threat in enumerate(threats):\n",
        "        analyst = analysts[idx % num_analysts]\n",
        "        prompt = remediation_instructions_template.format(\n",
        "            analyst_name=analyst.name,\n",
        "            role=analyst.role,\n",
        "            expertise=analyst.expertise,\n",
        "            threat_indicator=threat.indicator,\n",
        "            threat_type=threat.type\n",
        "        )\n",
        "        system_message = SystemMessage(content=prompt)\n",
        "        try:\n",
        "            response = llm.invoke([system_message])\n",
        "            remediation = response.content.strip()\n",
        "            remediation_steps.append(\n",
        "                f\"**{analyst.name} recommends:**\\n{remediation}\"\n",
        "            )\n",
        "        except Exception as e:\n",
        "            logging.error(f\"Error during remediation step generation for {threat.indicator}: {e}\")\n",
        "    logging.debug(\"Remediation steps generation completed.\")\n",
        "    return {'remediation_steps': remediation_steps}\n",
        "\n",
        "def compile_incident_report(state: DLPState):\n",
        "    analysis_sections = state.get('analysis_sections', [])\n",
        "    remediation_steps = state.get('remediation_steps', [])\n",
        "    logging.debug(\"Compiling incident report...\")\n",
        "    report = \"# Incident Report\\n\\n## Analysis\\n\\n\"\n",
        "    report += \"\\n\".join(analysis_sections)\n",
        "    report += \"\\n## Remediation Steps\\n\\n\"\n",
        "    report += \"\\n\".join(remediation_steps)\n",
        "    logging.debug(\"Incident report compiled.\")\n",
        "    return {'final_report': report}\n",
        "\n",
        "# Build the main graph\n",
        "from langgraph.graph import StateGraph, START, END\n",
        "from langgraph.checkpoint.memory import MemorySaver\n",
        "\n",
        "builder = StateGraph(DLPState)\n",
        "builder.add_node('extract_threats', extract_threats)\n",
        "builder.add_node('create_analysts', create_analysts)\n",
        "builder.add_node('analyze_threats', analyze_threats)\n",
        "builder.add_node('suggest_remediation_steps', suggest_remediation_steps)\n",
        "builder.add_node('compile_incident_report', compile_incident_report)\n",
        "\n",
        "# Edges\n",
        "builder.add_edge(START, 'extract_threats')\n",
        "builder.add_edge('extract_threats', 'create_analysts')\n",
        "builder.add_edge('create_analysts', 'analyze_threats')\n",
        "builder.add_edge('analyze_threats', 'suggest_remediation_steps')\n",
        "builder.add_edge('suggest_remediation_steps', 'compile_incident_report')\n",
        "builder.add_edge('compile_incident_report', END)\n",
        "\n",
        "# Compile the graph\n",
        "memory = MemorySaver()\n",
        "dlp_graph = builder.compile(checkpointer=memory)\n",
        "\n",
        "# Define the Gradio app function\n",
        "def run_dlp_app(incident_description, max_analysts):\n",
        "    # Initialize state\n",
        "    initial_state = {\n",
        "        'user_id': 'analyst1',\n",
        "        'incident_description': incident_description,\n",
        "        'messages': [],\n",
        "        'max_analysts': int(max_analysts)\n",
        "    }\n",
        "\n",
        "    # Config\n",
        "    config = {'configurable': {'thread_id': 'dlp_thread1'}}\n",
        "\n",
        "    logging.debug(\"Invoking the DLP graph...\")\n",
        "    # Execute the graph\n",
        "    try:\n",
        "        result_state = dlp_graph.invoke(initial_state, config)\n",
        "        logging.debug(\"Graph execution completed successfully.\")\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error during graph invocation: {e}\")\n",
        "        return \"An error occurred during the analysis. Please check the logs.\"\n",
        "\n",
        "    # Get the final state\n",
        "    try:\n",
        "        final_state = dlp_graph.get_state(config)[0]\n",
        "        # Return the final report\n",
        "        return final_state['final_report']\n",
        "    except Exception as e:\n",
        "        logging.error(f\"Error retrieving final state: {e}\")\n",
        "        return \"An error occurred while retrieving the final report. Please check the logs.\"\n",
        "\n",
        "# Create the Gradio interface\n",
        "iface = gr.Interface(\n",
        "    fn=run_dlp_app,\n",
        "    inputs=[\n",
        "        gr.Textbox(lines=10, label=\"Incident Description\", placeholder=\"Enter the incident description here...\"),\n",
        "        gr.Slider(minimum=1, maximum=5, value=3, step=1, label=\"Number of Analysts\")\n",
        "    ],\n",
        "    outputs=\"markdown\",\n",
        "    title=\"DLP Incident Analysis\",\n",
        "    description=\"Analyze cybersecurity incidents using AI-generated security analyst personas.\"\n",
        ")\n",
        "\n",
        "# Launch the Gradio app\n",
        "iface.launch(share=True)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 592
        },
        "id": "rla6U6StdRvw",
        "outputId": "7806e177-c42b-4b37-f51b-af5897b94602"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "* Running on public URL: https://edd17d7b8921ab44d8.gradio.live\n",
            "\n",
            "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://edd17d7b8921ab44d8.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "sdcazi95fOsm"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}