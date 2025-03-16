"""
Planning and Reasoning Module for Web Research Assistant
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import json
import time

# ---- Plan Data Models ----

class ResearchSubtopic(BaseModel):
    """A subtopic of the main research topic"""
    id: str
    title: str
    description: str
    priority: int = 1  # 1-5, with 5 being highest
    status: str = "planned"  # planned, in_progress, completed
    estimated_time: int = 15  # minutes
    search_queries: List[str] = []
    findings: List[str] = []
    conclusions: List[str] = []


class ResearchPlan(BaseModel):
    """Overall research plan"""
    id: str
    main_topic: str
    objective: str
    audience: str
    depth: str
    format: str
    time_allocation: Dict[str, int] = {}  # minutes per stage
    subtopics: List[ResearchSubtopic] = []
    status: str = "planning"  # planning, researching, synthesizing, reporting, completed
    start_time: float = Field(default_factory=time.time)
    deadline: Optional[float] = None
    
    def time_spent(self) -> int:
        """Return time spent in minutes"""
        return int((time.time() - self.start_time) / 60)
    
    def time_remaining(self) -> Optional[int]:
        """Return time remaining in minutes, if deadline exists"""
        if self.deadline:
            return max(0, int((self.deadline - time.time()) / 60))
        return None
    
    def progress(self) -> float:
        """Return overall progress as a percentage"""
        if not self.subtopics:
            return 0.0
        
        completed = sum(1 for st in self.subtopics if st.status == "completed")
        in_progress = sum(0.5 for st in self.subtopics if st.status == "in_progress")
        
        return 100 * (completed + in_progress) / len(self.subtopics)


# ---- Planner Implementation ----

class ResearchPlanner:
    """Handles planning and task management for research"""
    
    def __init__(self, llm_client, memory_manager):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.current_plan = None
    
    def create_initial_plan(self, topic: str, parameters: Dict[str, Any]) -> ResearchPlan:
        """Create initial research plan"""
        # Extract parameters
        audience = parameters.get("audience", "general")
        depth = parameters.get("depth", "medium")
        format = parameters.get("format", "report")
        time_limit = parameters.get("time_limit", 60)  # minutes
        
        # Generate plan using LLM
        prompt = f"""
        Create a detailed research plan for the topic: "{topic}"
        
        Audience: {audience}
        Depth: {depth}
        Format: {format}
        Time limit: {time_limit} minutes
        
        Your plan should include:
        1. Main objective of the research
        2. 3-7 key subtopics to investigate
        3. For each subtopic:
           - Clear title
           - Brief description
           - Priority (1-5, with 5 being highest)
           - 2-3 initial search queries
        4. Time allocation for different research phases
        
        Respond with a JSON object that follows this schema:
        {{
            "objective": "...",
            "time_allocation": {{ "searching": X, "reading": Y, "synthesizing": Z, "reporting": W }},
            "subtopics": [
                {{
                    "id": "subtopic-1",
                    "title": "...",
                    "description": "...",
                    "priority": 5,
                    "search_queries": ["...", "..."]
                }},
                ...
            ]
        }}
        """
        
        response = self.llm_client.generate(prompt, max_tokens=2000, temperature=0.2)
        
        try:
            # Parse LLM response
            plan_data = json.loads(response)
            
            # Create plan object
            plan = ResearchPlan(
                id=f"plan-{int(time.time())}",
                main_topic=topic,
                objective=plan_data.get("objective", f"Research {topic}"),
                audience=audience,
                depth=depth,
                format=format,
                time_allocation=plan_data.get("time_allocation", {
                    "searching": time_limit * 0.2,
                    "reading": time_limit * 0.3,
                    "synthesizing": time_limit * 0.3,
                    "reporting": time_limit * 0.2
                })
            )
            
            # Set deadline if time limit provided
            if time_limit:
                plan.deadline = time.time() + (time_limit * 60)
            
            # Add subtopics
            for st_data in plan_data.get("subtopics", []):
                subtopic = ResearchSubtopic(
                    id=st_data.get("id", f"st-{len(plan.subtopics)+1}"),
                    title=st_data.get("title", "Untitled Subtopic"),
                    description=st_data.get("description", ""),
                    priority=st_data.get("priority", 3),
                    search_queries=st_data.get("search_queries", [])
                )
                plan.subtopics.append(subtopic)
            
            # Sort subtopics by priority
            plan.subtopics.sort(key=lambda x: x.priority, reverse=True)
            
            self.current_plan = plan
            return plan
            
        except Exception as e:
            # Fallback plan if parsing fails
            plan = ResearchPlan(
                id=f"plan-{int(time.time())}",
                main_topic=topic,
                objective=f"Research {topic} comprehensively",
                audience=audience,
                depth=depth,
                format=format,
                time_allocation={
                    "searching": time_limit * 0.2,
                    "reading": time_limit * 0.3,
                    "synthesizing": time_limit * 0.3,
                    "reporting": time_limit * 0.2
                },
                subtopics=[
                    ResearchSubtopic(
                        id="st-1",
                        title="General Overview",
                        description=f"General information about {topic}",
                        priority=5,
                        search_queries=[f"{topic} overview", f"{topic} introduction", f"{topic} basics"]
                    )
                ]
            )
            
            if time_limit:
                plan.deadline = time.time() + (time_limit * 60)
            
            self.current_plan = plan
            return plan
    
    def refine_plan(self, feedback: str) -> ResearchPlan:
        """Refine research plan based on feedback"""
        if not self.current_plan:
            raise ValueError("No current plan to refine")
        
        # Create prompt for refinement
        prompt = f"""
        Refine the current research plan based on the following feedback:
        "{feedback}"
        
        Current plan:
        - Main topic: {self.current_plan.main_topic}
        - Objective: {self.current_plan.objective}
        - Subtopics:
          {self._format_subtopics_for_prompt()}
        
        Suggest specific changes to the plan. Respond with a JSON object that includes:
        1. New or modified subtopics
        2. Subtopics to remove (if any)
        3. Changes to priorities
        4. Additional search queries
        
        JSON schema:
        {{
            "modified_objective": "...",  # Optional
            "add_subtopics": [
                {{
                    "id": "new-subtopic-1",
                    "title": "...",
                    "description": "...",
                    "priority": 4,
                    "search_queries": ["...", "..."]
                }}
            ],
            "remove_subtopics": ["subtopic-id-1", "subtopic-id-2"],  # IDs to remove
            "modify_subtopics": [
                {{
                    "id": "existing-subtopic-id",
                    "title": "...",  # Optional
                    "description": "...",  # Optional
                    "priority": 5,  # Optional
                    "search_queries": ["...", "..."]  # Optional, will be added to existing
                }}
            ]
        }}
        """
        
        response = self.llm_client.generate(prompt, max_tokens=2000, temperature=0.2)
        
        try:
            # Parse LLM response
            changes = json.loads(response)
            
            # Apply changes to plan
            if "modified_objective" in changes:
                self.current_plan.objective = changes["modified_objective"]
            
            # Remove subtopics
            if "remove_subtopics" in changes:
                self.current_plan.subtopics = [
                    st for st in self.current_plan.subtopics 
                    if st.id not in changes["remove_subtopics"]
                ]
            
            # Modify existing subtopics
            if "modify_subtopics" in changes:
                for mod in changes["modify_subtopics"]:
                    for st in self.current_plan.subtopics:
                        if st.id == mod["id"]:
                            if "title" in mod:
                                st.title = mod["title"]
                            if "description" in mod:
                                st.description = mod["description"]
                            if "priority" in mod:
                                st.priority = mod["priority"]
                            if "search_queries" in mod:
                                st.search_queries.extend(mod["search_queries"])
            
            # Add new subtopics
            if "add_subtopics" in changes:
                for st_data in changes["add_subtopics"]:
                    subtopic = ResearchSubtopic(
                        id=st_data.get("id", f"st-{int(time.time())}-{len(self.current_plan.subtopics)}"),
                        title=st_data.get("title", "Untitled Subtopic"),
                        description=st_data.get("description", ""),
                        priority=st_data.get("priority", 3),
                        search_queries=st_data.get("search_queries", [])
                    )
                    self.current_plan.subtopics.append(subtopic)
            
            # Re-sort subtopics by priority
            self.current_plan.subtopics.sort(key=lambda x: x.priority, reverse=True)
            
            return self.current_plan
            
        except Exception as e:
            # Return current plan if parsing fails
            return self.current_plan
    
    def get_next_task(self) -> Tuple[str, Dict[str, Any]]:
        """Get the next research task based on current plan and state"""
        if not self.current_plan:
            return "plan", {"message": "No research plan exists. Create a plan first."}
        
        # Check plan status
        if self.current_plan.status == "planning":
            return "start_research", {"plan_id": self.current_plan.id}
        
        if self.current_plan.status == "completed":
            return "report", {"plan_id": self.current_plan.id}
        
        # Find next subtopic to work on
        for subtopic in self.current_plan.subtopics:
            if subtopic.status == "planned":
                return "research_subtopic", {
                    "subtopic_id": subtopic.id,
                    "title": subtopic.title,
                    "description": subtopic.description,
                    "search_queries": subtopic.search_queries
                }
        
        # If all subtopics are in progress or completed, check if any are in progress
        for subtopic in self.current_plan.subtopics:
            if subtopic.status == "in_progress":
                return "continue_subtopic", {
                    "subtopic_id": subtopic.id,
                    "title": subtopic.title,
                    "description": subtopic.description
                }
        
        # If all subtopics are completed, move to synthesis
        if self.current_plan.status == "researching":
            self.current_plan.status = "synthesizing"
            return "synthesize", {"plan_id": self.current_plan.id}
        
        if self.current_plan.status == "synthesizing":
            self.current_plan.status = "reporting"
            return "create_report", {"plan_id": self.current_plan.id}
        
        # Default
        return "assess_progress", {"plan_id": self.current_plan.id}
    
    def update_subtopic_status(self, subtopic_id: str, status: str, findings: List[str] = None) -> bool:
        """Update the status of a subtopic"""
        if not self.current_plan:
            return False
        
        for subtopic in self.current_plan.subtopics:
            if subtopic.id == subtopic_id:
                subtopic.status = status
                if findings:
                    subtopic.findings.extend(findings)
                return True
        
        return False
    
    def add_subtopic_conclusion(self, subtopic_id: str, conclusion: str) -> bool:
        """Add a conclusion to a subtopic"""
        if not self.current_plan:
            return False
        
        for subtopic in self.current_plan.subtopics:
            if subtopic.id == subtopic_id:
                subtopic.conclusions.append(conclusion)
                return True
        
        return False
    
    def update_plan_status(self, status: str) -> bool:
        """Update the overall plan status"""
        if not self.current_plan:
            return False
        
        self.current_plan.status = status
        return True
    
    def generate_progress_report(self) -> Dict[str, Any]:
        """Generate a progress report for the current plan"""
        if not self.current_plan:
            return {"error": "No active research plan"}
        
        # Calculate statistics
        total_subtopics = len(self.current_plan.subtopics)
        completed_subtopics = sum(1 for st in self.current_plan.subtopics if st.status == "completed")
        in_progress_subtopics = sum(1 for st in self.current_plan.subtopics if st.status == "in_progress")
        planned_subtopics = sum(1 for st in self.current_plan.subtopics if st.status == "planned")
        
        total_findings = sum(len(st.findings) for st in self.current_plan.subtopics)
        total_conclusions = sum(len(st.conclusions) for st in self.current_plan.subtopics)
        
        time_spent = self.current_plan.time_spent()
        time_remaining = self.current_plan.time_remaining()
        
        return {
            "plan_id": self.current_plan.id,
            "main_topic": self.current_plan.main_topic,
            "status": self.current_plan.status,
            "progress_percentage": self.current_plan.progress(),
            "time_spent_minutes": time_spent,
            "time_remaining_minutes": time_remaining,
            "subtopics_stats": {
                "total": total_subtopics,
                "completed": completed_subtopics,
                "in_progress": in_progress_subtopics,
                "planned": planned_subtopics
            },
            "findings_count": total_findings,
            "conclusions_count": total_conclusions,
            "next_steps": self._generate_next_steps()
        }
    
    def _generate_next_steps(self) -> List[str]:
        """Generate list of next steps based on current state"""
        next_steps = []
        
        if not self.current_plan:
            return ["Create a research plan"]
        
        if self.current_plan.status == "planning":
            next_steps.append("Start research phase")
        
        # Add steps for incomplete subtopics
        planned_subtopics = [st for st in self.current_plan.subtopics if st.status == "planned"]
        in_progress_subtopics = [st for st in self.current_plan.subtopics if st.status == "in_progress"]
        
        for st in in_progress_subtopics[:2]:  # Show max 2
            next_steps.append(f"Continue research on subtopic: {st.title}")
        
        for st in planned_subtopics[:3]:  # Show max 3
            next_steps.append(f"Begin research on subtopic: {st.title}")
        
        if self.current_plan.status == "synthesizing":
            next_steps.append("Synthesize research findings into coherent conclusions")
        
        if self.current_plan.status == "reporting":
            next_steps.append("Create final research report")
        
        return next_steps
    
    def _format_subtopics_for_prompt(self) -> str:
        """Format subtopics for inclusion in prompts"""
        if not self.current_plan:
            return ""
        
        result = ""
        for st in self.current_plan.subtopics:
            result += f"    * {st.title} (ID: {st.id}, Priority: {st.priority})\n"
            result += f"      Description: {st.description}\n"
            result += f"      Status: {st.status}\n"
            if st.search_queries:
                result += f"      Queries: {', '.join(st.search_queries)}\n"
            result += "\n"
        
        return result


# ---- Reasoning Engine ----

class ReasoningEngine:
    """Handles complex reasoning for research and synthesis"""
    
    def __init__(self, llm_client, memory_manager):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
    
    def analyze_search_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze search results for relevance and credibility"""
        # Format results for the prompt
        results_text = ""
        for i, result in enumerate(results):
            results_text += f"{i+1}. {result.get('title', 'Untitled')}\n"
            results_text += f"   URL: {result.get('url', 'No URL')}\n"
            results_text += f"   Snippet: {result.get('snippet', 'No snippet')}\n\n"
        
        prompt = f"""
        Analyze the following search results for the query: "{query}"
        
        {results_text}
        
        For each result, determine:
        1. Relevance score (0-100) to the query
        2. Credibility assessment (high, medium, low)
        3. Whether to prioritize it for detailed examination
        
        Respond with a JSON array of objects, one for each result:
        [
            {{
                "result_index": 1,
                "relevance_score": 85,
                "credibility": "high",
                "priority": true,
                "reasoning": "Brief explanation of your assessment"
            }},
            ...
        ]
        """
        
        response = self.llm_client.generate(prompt, max_tokens=2000, temperature=0.1)
        
        try:
            # Parse LLM response
            analysis = json.loads(response)
            
            # Match analysis back to original results
            enriched_results = []
            for result in results:
                result_index = results.index(result)
                
                # Find corresponding analysis
                for item in analysis:
                    if item.get("result_index") == result_index + 1:
                        result.update({
                            "relevance_score": item.get("relevance_score", 50),
                            "credibility": item.get("credibility", "medium"),
                            "priority": item.get("priority", False),
                            "analysis_reasoning": item.get("reasoning", "")
                        })
                        break
                
                enriched_results.append(result)
            
            # Sort by relevance score and priority
            enriched_results.sort(key=lambda x: (x.get("priority", False), x.get("relevance_score", 0)), reverse=True)
            
            return enriched_results
            
        except Exception as e:
            # Return original results if parsing fails
            return results
    
    def extract_facts(self, content: str, subtopic: str) -> List[Dict[str, Any]]:
        """Extract key facts from content"""
        prompt = f"""
        Extract key facts from the following content related to the subtopic: "{subtopic}"
        
        CONTENT:
        {content[:4000]}  # Limit content length
        
        Extract 5-10 specific, well-defined facts that are:
        1. Directly supported by the content
        2. Relevant to the subtopic
        3. Specific and precise (not vague generalizations)
        4. Independent of each other
        
        For each fact, assign a confidence score (0-100) based on how clearly it is supported by the content.
        
        Respond with a JSON array of fact objects:
        [
            {{
                "fact": "Specific factual statement",
                "confidence": 95,
                "quote": "Direct quote from the content that supports this fact"
            }},
            ...
        ]
        """
        
        response = self.llm_client.generate(prompt, max_tokens=2000, temperature=0.1)
        
        try:
            # Parse LLM response
            facts = json.loads(response)
            return facts
        except Exception as e:
            # Return empty list if parsing fails
            return []
    
    def synthesize_subtopic(self, subtopic: str, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize findings into cohesive conclusions for a subtopic"""
        # Format facts for the prompt
        facts_text = ""
        for i, fact in enumerate(facts):
            facts_text += f"{i+1}. {fact.get('fact', 'No fact provided')}\n"
            facts_text += f"   Confidence: {fact.get('confidence', 0)}%\n"
            if 'quote' in fact:
                facts_text += f"   Support: {fact.get('quote', '')}\n"
            facts_text += "\n"
        
        prompt = f"""
        Synthesize the following facts about the subtopic: "{subtopic}"
        
        FACTS:
        {facts_text}
        
        Create a coherent synthesis that:
        1. Presents key findings about the subtopic
        2. Identifies patterns, trends, or relationships between facts
        3. Acknowledges any contradictions or uncertainties
        4. Creates 1-3 clear conclusions
        
        Respond with a JSON object:
        {{
            "synthesis": "Comprehensive paragraph synthesizing the facts",
            "key_points": [
                "Key point 1",
                "Key point 2",
                ...
            ],
            "conclusions": [
                {{
                    "conclusion": "Clear conclusion statement",
                    "confidence": 85,
                    "supporting_facts": [1, 4, 7]  # Indices of supporting facts
                }},
                ...
            ],
            "gaps": [
                "Identified gap in the research 1",
                ...
            ]
        }}
        """
        
        response = self.llm_client.generate(prompt, max_tokens=2000, temperature=0.2)
        
        try:
            # Parse LLM response
            synthesis = json.loads(response)
            return synthesis
        except Exception as e:
            # Return basic structure if parsing fails
            return {
                "synthesis": f"Synthesis of facts about {subtopic}",
                "key_points": [f.get("fact") for f in facts[:3] if "fact" in f],
                "conclusions": [],
                "gaps": ["Unable to identify specific research gaps"]
            }
    
    def generate_report_outline(self, topic: str, subtopics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate outline for the final report"""
        # Format subtopics for the prompt
        subtopics_text = ""
        for i, st in enumerate(subtopics):
            subtopics_text += f"{i+1}. {st.get('title', 'Untitled')}\n"
            subtopics_text += f"   Key points: {', '.join(st.get('key_points', []))}\n"
            subtopics_text += f"   Main conclusions: {', '.join([c.get('conclusion', '') for c in st.get('conclusions', [])])}\n\n"
        
        prompt = f"""
        Create a detailed outline for a research report on: "{topic}"
        
        The report should cover the following subtopics:
        
        {subtopics_text}
        
        Create a comprehensive report outline that includes:
        1. Executive Summary
        2. Introduction
        3. Methodology
        4. Main sections (organized logically)
        5. Conclusions and implications
        6. References
        
        Respond with a JSON object:
        {{
            "title": "Suggested report title",
            "sections": [
                {{
                    "title": "Section title",
                    "subsections": [
                        {{
                            "title": "Subsection title",
                            "content_description": "Brief description of content to include",
                            "estimated_length": "1-2 paragraphs"
                        }},
                        ...
                    ]
                }},
                ...
            ]
        }}
        """
        
        response = self.llm_client.generate(prompt, max_tokens=2000, temperature=0.3)
        
        try:
            # Parse LLM response
            outline = json.loads(response)
            return outline
        except Exception as e:
            # Return basic outline if parsing fails
            return {
                "title": f"Research Report: {topic}",
                "sections": [
                    {
                        "title": "Executive Summary",
                        "subsections": [
                            {
                                "title": "Key Findings",
                                "content_description": "Summary of main findings",
                                "estimated_length": "1-2 paragraphs"
                            }
                        ]
                    },
                    {
                        "title": "Introduction",
                        "subsections": [
                            {
                                "title": "Background",
                                "content_description": "Context of the research",
                                "estimated_length": "1-2 paragraphs"
                            }
                        ]
                    },
                    {
                        "title": "Findings",
                        "subsections": [
                            {
                                "title": st.get("title", "Untitled Section"),
                                "content_description": "Findings related to this subtopic",
                                "estimated_length": "2-3 paragraphs"
                            } for st in subtopics
                        ]
                    },
                    {
                        "title": "Conclusions",
                        "subsections": [
                            {
                                "title": "Key Conclusions",
                                "content_description": "Major conclusions from the research",
                                "estimated_length": "1-2 paragraphs"
                            }
                        ]
                    }
                ]
            }
    
    def evaluate_source_conflict(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between contradictory facts from different sources"""
        prompt = f"""
        Evaluate the following conflicting facts:
        
        FACT 1: {fact1.get('fact', 'No fact provided')}
        Confidence: {fact1.get('confidence', 0)}%
        Source: {fact1.get('source_url', 'Unknown')}
        
        FACT 2: {fact2.get('fact', 'No fact provided')}
        Confidence: {fact2.get('confidence', 0)}%
        Source: {fact2.get('source_url', 'Unknown')}
        
        These facts appear to contradict each other. Please:
        1. Identify the specific contradiction
        2. Evaluate the credibility of each source
        3. Consider possible explanations for the contradiction
        4. Make a determination about which fact is more likely to be accurate, or whether additional research is needed
        
        Respond with a JSON object:
        {{
            "contradiction": "Description of the specific contradiction",
            "source1_credibility": "Assessment of the first source's credibility",
            "source2_credibility": "Assessment of the second source's credibility",
            "possible_explanations": [
                "Possible explanation 1",
                "Possible explanation 2"
            ],
            "determination": "More reliable fact or 'inconclusive'",
            "confidence": 75,  # 0-100 confidence in your determination
            "recommended_action": "Specific recommendation for further research or reporting"
        }}
        """
        
        response = self.llm_client.generate(prompt, max_tokens=1500, temperature=0.2)
        
        try:
            # Parse LLM response
            evaluation = json.loads(response)
            return evaluation
        except Exception as e:
            # Return basic evaluation if parsing fails
            return {
                "contradiction": f"Contradiction between '{fact1.get('fact', '')}' and '{fact2.get('fact', '')}'",
                "determination": "inconclusive",
                "confidence": 50,
                "recommended_action": "Seek additional sources to resolve this contradiction"
            }
    
    def generate_critical_questions(self, topic: str, current_findings: List[Dict[str, Any]]) -> List[str]:
        """Generate critical questions to guide further research"""
        # Format current findings for the prompt
        findings_text = ""
        for i, finding in enumerate(current_findings[:10]):  # Limit to 10 findings
            findings_text += f"{i+1}. {finding.get('fact', 'No fact provided')}\n"
        
        prompt = f"""
        Based on the current research findings on "{topic}", generate critical questions to guide further research.
        
        CURRENT FINDINGS:
        {findings_text}
        
        Generate 5-7 critical questions that:
        1. Address important gaps in the current findings
        2. Challenge potential assumptions
        3. Explore deeper implications
        4. Consider alternative perspectives
        5. Target areas where more evidence is needed
        
        Each question should be specific, focused, and designed to substantially advance the research.
        
        Respond with a JSON array of question objects:
        [
            {{
                "question": "Specific critical question",
                "rationale": "Brief explanation of why this question is important",
                "search_queries": ["Suggested search query 1", "Suggested search query 2"]
            }},
            ...
        ]
        """
        
        response = self.llm_client.generate(prompt, max_tokens=1500, temperature=0.4)
        
        try:
            # Parse LLM response
            questions = json.loads(response)
            return questions
        except Exception as e:
            # Return basic questions if parsing fails
            return [
                {
                    "question": f"What are the main challenges related to {topic}?",
                    "rationale": "Understanding challenges provides critical context",
                    "search_queries": [f"{topic} challenges", f"{topic} problems"]
                },
                {
                    "question": f"What are the latest developments in {topic}?",
                    "rationale": "Ensuring research captures recent information",
                    "search_queries": [f"{topic} latest developments", f"{topic} recent advances"]
                }
            ]