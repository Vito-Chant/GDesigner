"""
CoRe Framework: Handoff Generator Module
Generates explicit handoff notes for agent-to-agent transitions
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio


@dataclass
class HandoffNote:
    """Structured handoff information"""
    from_agent: str
    to_agent: str
    completed_work: str
    current_progress: str
    next_steps: str
    warnings: List[str]
    context_summary: str
    confidence_in_handoff: float

    def to_text(self) -> str:
        """Convert to natural language for next agent"""
        note = f"""
=== HANDOFF NOTE ===
From: {self.from_agent}
To: {self.to_agent}

What I've Done:
{self.completed_work}

Current Progress:
{self.current_progress}

What You Need to Do:
{self.next_steps}
"""
        if self.warnings:
            note += f"\n⚠️ Important Warnings:\n"
            for warning in self.warnings:
                note += f"- {warning}\n"

        note += f"\nContext Summary: {self.context_summary}\n"
        note += f"My Confidence in This Handoff: {self.confidence_in_handoff:.2f}\n"
        note += "==================\n"

        return note


class HandoffGenerator:
    """
    Generates explicit, structured handoff notes for agent transitions

    This is a key innovation over AnyMAC's NCS which only concatenates history
    """

    def __init__(self, llm):
        self.llm = llm

    async def generate_handoff_note(
            self,
            current_agent: str,
            next_agent: str,
            original_task: str,
            current_agent_output: str,
            full_history: List[Dict],
            next_agent_profile: str,
            routing_reasoning: str
    ) -> HandoffNote:
        """
        Generate a structured handoff note

        Args:
            current_agent: ID of current agent
            next_agent: ID of next agent
            original_task: The original user task
            current_agent_output: What current agent produced
            full_history: Complete interaction history
            next_agent_profile: Profile of next agent
            routing_reasoning: Why next agent was selected

        Returns:
            HandoffNote with structured information
        """

        # Build prompt for handoff generation
        prompt = self._build_handoff_prompt(
            current_agent, next_agent, original_task,
            current_agent_output, full_history,
            next_agent_profile, routing_reasoning
        )

        messages = [
            {
                'role': 'system',
                'content': 'You are an expert at task delegation. Generate clear, structured handoff notes.'
            },
            {'role': 'user', 'content': prompt}
        ]

        response = await self.llm.agen(messages)

        # Parse response into structured HandoffNote
        handoff = self._parse_handoff_response(
            response, current_agent, next_agent
        )

        return handoff

    def _build_handoff_prompt(
            self,
            current_agent: str,
            next_agent: str,
            original_task: str,
            current_output: str,
            history: List[Dict],
            next_profile: str,
            routing_reason: str
    ) -> str:
        """Build prompt for LLM to generate handoff"""

        prompt = f"""You are {current_agent} handing off work to {next_agent}.

ORIGINAL TASK:
{original_task}

YOUR OUTPUT:
{current_output}

NEXT AGENT'S PROFILE:
{next_profile}

WHY THEY WERE SELECTED:
{routing_reason}
"""

        if history:
            prompt += "\nPREVIOUS WORK (History):\n"
            for i, step in enumerate(history[-3:], 1):  # Last 3 steps
                agent = step.get('agent', 'Unknown')
                output = step.get('output', '')[:200]  # Truncate
                prompt += f"{i}. {agent}: {output}...\n"

        prompt += """
GENERATE A HANDOFF NOTE with these sections:

COMPLETED_WORK: What you have accomplished
CURRENT_PROGRESS: What state the task is in now
NEXT_STEPS: Specific instructions for what {next_agent} should do
WARNINGS: Any pitfalls or issues they should watch out for (list 0-3)
CONTEXT_SUMMARY: Brief summary of key context they need
CONFIDENCE: Your confidence this handoff will succeed (0.0-1.0)

Format your response as:
COMPLETED_WORK: <text>
CURRENT_PROGRESS: <text>
NEXT_STEPS: <text>
WARNINGS: <warning1>|<warning2>|... (use | as separator, or NONE)
CONTEXT_SUMMARY: <text>
CONFIDENCE: <number>
"""

        return prompt

    def _parse_handoff_response(
            self,
            response: str,
            from_agent: str,
            to_agent: str
    ) -> HandoffNote:
        """Parse LLM response into HandoffNote structure"""

        lines = response.strip().split('\n')

        # Default values
        completed_work = "Work completed by previous agent"
        current_progress = "In progress"
        next_steps = "Continue working on the task"
        warnings = []
        context_summary = "See previous outputs"
        confidence = 0.7

        # Parse each section
        for line in lines:
            line = line.strip()

            if line.startswith('COMPLETED_WORK:'):
                completed_work = line.replace('COMPLETED_WORK:', '').strip()
            elif line.startswith('CURRENT_PROGRESS:'):
                current_progress = line.replace('CURRENT_PROGRESS:', '').strip()
            elif line.startswith('NEXT_STEPS:'):
                next_steps = line.replace('NEXT_STEPS:', '').strip()
            elif line.startswith('WARNINGS:'):
                warnings_str = line.replace('WARNINGS:', '').strip()
                if warnings_str and warnings_str.upper() != 'NONE':
                    warnings = [w.strip() for w in warnings_str.split('|') if w.strip()]
            elif line.startswith('CONTEXT_SUMMARY:'):
                context_summary = line.replace('CONTEXT_SUMMARY:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                except:
                    confidence = 0.7

        return HandoffNote(
            from_agent=from_agent,
            to_agent=to_agent,
            completed_work=completed_work,
            current_progress=current_progress,
            next_steps=next_steps,
            warnings=warnings,
            context_summary=context_summary,
            confidence_in_handoff=confidence
        )

    async def generate_simple_handoff(
            self,
            from_agent: str,
            to_agent: str,
            task_description: str,
            output: str
    ) -> str:
        """
        Quick handoff generation without full LLM call
        Used when fast handoff is needed
        """

        return f"""From {from_agent} to {to_agent}:

Task: {task_description}
My output: {output}

Please continue from here.
"""


# Example usage
if __name__ == "__main__":
    class MockLLM:
        async def agen(self, messages):
            return """COMPLETED_WORK: I have analyzed the mathematical problem and identified it as a quadratic equation: ax^2 + bx + c = 0
CURRENT_PROGRESS: The equation structure is clear, but needs implementation
NEXT_STEPS: Implement a Python function that takes coefficients a, b, c as input and returns the two solutions using the quadratic formula
WARNINGS: Watch out for division by zero when a=0|Handle complex solutions when discriminant is negative
CONTEXT_SUMMARY: Standard quadratic equation solving task, no edge cases in the original problem
CONFIDENCE: 0.85"""


    async def test_handoff():
        llm = MockLLM()
        generator = HandoffGenerator(llm)

        handoff = await generator.generate_handoff_note(
            current_agent="math_analyzer_1",
            next_agent="code_expert_1",
            original_task="Solve the equation 2x^2 + 5x - 3 = 0",
            current_agent_output="This is a quadratic equation with a=2, b=5, c=-3",
            full_history=[],
            next_agent_profile="Expert programmer specialized in numerical algorithms",
            routing_reasoning="Selected for implementation expertise"
        )

        print(handoff.to_text())
        print(f"\nWarnings detected: {len(handoff.warnings)}")
        print(f"Confidence: {handoff.confidence_in_handoff:.2f}")


    asyncio.run(test_handoff())