"""
Sentiment analysis and question identification module
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import json


@dataclass
class ConversationAnalysis:
    """Analysis results for a conversation segment"""
    sentiment: str  # positive, negative, neutral, questioning
    is_question: bool
    question_type: Optional[str]  # clarification, objection, interest, technical
    detected_concerns: List[str]
    emotional_state: str  # engaged, skeptical, interested, resistant
    requires_response: bool
    is_complex_question: bool
    key_topics: List[str]


class SentimentAnalyzer:
    """Analyzes sentiment and identifies questions in real-time conversation"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
    async def analyze(self, transcript: str, history: List[Dict]) -> ConversationAnalysis:
        """Analyze transcript for sentiment, questions, and conversation dynamics"""
        
        # Build context from history
        context = self._build_context(history[-5:])  # Last 5 exchanges
        
        # Use reasoning model for deep analysis
        prompt = f"""Analyze this conversation segment for persuasion context:

Recent context:
{context}

Current statement: "{transcript}"

Provide analysis in JSON format:
{{
    "sentiment": "positive/negative/neutral/questioning",
    "is_question": true/false,
    "question_type": "clarification/objection/interest/technical" or null,
    "detected_concerns": ["list of concerns"],
    "emotional_state": "engaged/skeptical/interested/resistant",
    "requires_response": true/false,
    "is_complex_question": true/false,
    "key_topics": ["relevant topics mentioned"]
}}

Focus on identifying:
1. Questions that are critical to achieving the persuasive goal
2. Objections or concerns, which may be obstacles to the persuasive goal
3. Indicators of interest
4. Changes in sentiment or emotional state
5. Topics that relate concretely to persuasion objectives"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.reasoning_model,
                messages=[
                    {"role": "system", "content": "You are an expert at conversation analysis for persuasion."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=256
            )
            
            # Parse JSON response
            analysis_data = json.loads(response.choices[0].message.content)
            
            return ConversationAnalysis(
                sentiment=analysis_data.get("sentiment", "neutral"),
                is_question=analysis_data.get("is_question", False),
                question_type=analysis_data.get("question_type"),
                detected_concerns=analysis_data.get("detected_concerns", []),
                emotional_state=analysis_data.get("emotional_state", "neutral"),
                requires_response=analysis_data.get("requires_response", False),
                is_complex_question=analysis_data.get("is_complex_question", False),
                key_topics=analysis_data.get("key_topics", [])
            )
            
        except Exception as e:
            print(f"Analysis error: {e}")
            # Return default analysis on error
            return ConversationAnalysis(
                sentiment="neutral",
                is_question=self._simple_question_detection(transcript),
                question_type=None,
                detected_concerns=[],
                emotional_state="neutral",
                requires_response=self._simple_question_detection(transcript),
                is_complex_question=False,
                key_topics=[]
            )
            
    def _build_context(self, history: List[Dict]) -> str:
        """Build context string from conversation history"""
        if not history:
            return "No previous context."
            
        context_lines = []
        for exchange in history:
            if 'user' in exchange:
                context_lines.append(f"User: {exchange['user']}")
            if 'assistant' in exchange:
                context_lines.append(f"Assistant: {exchange['assistant']}")
                
        return "\n".join(context_lines)
        
    def _simple_question_detection(self, text: str) -> bool:
        """Simple fallback question detection"""
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'could', 'would', 'should']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)


class QuestionPredictor:
    """Predicts likely follow-up questions based on conversation flow"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
    async def predict_questions(self, 
                               current_topic: str, 
                               objective: 'PersuasionObjective',
                               history: List[Dict]) -> List[Dict[str, str]]:
        """Predict likely questions and prepare responses"""
        
        prompt = f"""Based on this persuasion scenario, predict likely questions:

Current topic: {current_topic}
Persuasion goal: {objective.main_goal}
Key points to cover: {', '.join(objective.key_points)}

Recent conversation:
{self._build_context(history[-3:])}

List 3-5 most likely follow-up questions the user might ask, along with the key points for answering each.

Format as JSON:
[
    {{
        "question": "predicted question",
        "answer_points": ["key point 1", "key point 2"],
        "probability": "high/medium/low"
    }}
]"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.reasoning_model,
                messages=[
                    {"role": "system", "content": "You predict conversation flow in sales and persuasion contexts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            
            predictions = json.loads(response.choices[0].message.content)
            return predictions
            
        except Exception as e:
            print(f"Question prediction error: {e}")
            return []
            
    def _build_context(self, history: List[Dict]) -> str:
        """Build context string from conversation history"""
        if not history:
            return "No previous context."
            
        context_lines = []
        for exchange in history:
            if 'user' in exchange:
                context_lines.append(f"User: {exchange['user']}")
            if 'assistant' in exchange:
                context_lines.append(f"Assistant: {exchange['assistant']}")
                
        return "\n".join(context_lines)
