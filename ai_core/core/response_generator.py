"""
Response generator with prosody integration
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
import json
import asyncio

from .prosody_tokenizer import ProsodyTokenizer, ProsodyStrategy


@dataclass
class PersuasiveResponse:
    """Generated response with prosody and metadata"""
    text: str
    prosody_text: str
    prosody_tokens: List[int]
    voice_profile: str
    confidence_score: float
    persuasion_tactics: List[str]


class ResponseGenerator:
    """Generates persuasive responses with prosody markup"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.prosody_tokenizer = ProsodyTokenizer()
        self.response_cache = {}  # Cache for predicted responses
        
    async def generate(self,
                      transcript: str,
                      analysis: 'ConversationAnalysis',
                      alignment: 'AlignmentAnalysis',
                      objective: 'PersuasionObjective',
                      history: List[Dict]) -> PersuasiveResponse:
        """Generate persuasive response with prosody"""
        
        # Check cache first (for predicted questions)
        cache_key = self._get_cache_key(transcript, analysis)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
            
        # Build comprehensive prompt
        prompt = self._build_response_prompt(
            transcript, analysis, alignment, objective, history
        )
        
        # Generate response using reasoning model
        response_text = await self._generate_text_response(prompt)
        
        # Apply prosody based on context
        prosody_context = self._build_prosody_context(analysis, alignment)
        prosody_text = self._apply_prosody(response_text, prosody_context)
        
        # Encode with prosody tokens
        prosody_tokens = self.prosody_tokenizer.encode_with_prosody(prosody_text)
        
        # Select appropriate voice
        voice_profile = self._select_voice(analysis, objective)
        
        # Identify tactics used
        tactics = self._identify_tactics(response_text, alignment)
        
        response = PersuasiveResponse(
            text=response_text,
            prosody_text=prosody_text,
            prosody_tokens=prosody_tokens,
            voice_profile=voice_profile,
            confidence_score=self._calculate_confidence(alignment),
            persuasion_tactics=tactics
        )
        
        return response
        
    def _build_response_prompt(self,
                              transcript: str,
                              analysis: 'ConversationAnalysis',
                              alignment: 'AlignmentAnalysis',
                              objective: 'PersuasionObjective',
                              history: List[Dict]) -> str:
        """Build comprehensive prompt for response generation"""
        
        # Get recent context
        context = self._format_history(history[-3:])
        
        prompt = f"""Generate a persuasive response for this scenario:

OBJECTIVE: {objective.main_goal}
KEY POINTS TO COVER: {', '.join(objective.key_points)}
AUDIENCE TRIGGERS: {', '.join(objective.audience_triggers)}

CURRENT STATEMENT: "{transcript}"
ANALYSIS:
- Type: {analysis.question_type if analysis.is_question else 'statement'}
- Sentiment: {analysis.sentiment}
- Emotional State: {analysis.emotional_state}
- Concerns: {', '.join(analysis.detected_concerns)}

STRATEGIC CONTEXT:
- Alignment Score: {alignment.alignment_score:.2f}
- Remaining Points: {', '.join(alignment.remaining_points[:3])}
- Opportunities: {', '.join(alignment.detected_opportunities)}
- Suggested Action: {alignment.next_best_action}

RECENT CONVERSATION:
{context}

Generate a persuasive response that:
1. Directly addresses any questions or concerns
2. Advances toward the objective
3. Uses audience triggers naturally
4. Maintains appropriate emotional tone
5. Creates forward momentum

Response should be natural, conversational, and persuasive without being pushy.
Maximum 2-3 sentences unless answering a complex question."""

        return prompt
        
    async def _generate_text_response(self, prompt: str) -> str:
        """Generate text response using reasoning model"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.reasoning_model,
                messages=[
                    {"role": "system", "content": "You are a persuasive communication expert. Generate natural, compelling responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=256
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I understand your point. Let me address that for you."
            
    def _apply_prosody(self, text: str, context: Dict) -> str:
        """Apply prosody markup to response text"""
        
        # Use ProsodyStrategy for intelligent prosody application
        prosody_text = ProsodyStrategy.apply_persuasion_prosody(text, context)
        
        # Additional context-specific prosody
        if context.get('is_addressing_concern'):
            # Add calming prosody for concerns
            prosody_text = f"<CALM> {prosody_text}"
            
        if context.get('is_presenting_benefit'):
            # Add enthusiasm for benefits
            prosody_text = prosody_text.replace("save", "<EMPH>save")
            prosody_text = prosody_text.replace("improve", "<EMPH>improve")
            prosody_text = prosody_text.replace("increase", "<EMPH>increase")
            
        if context.get('urgency_level') == 'high':
            # Add urgency markers
            prosody_text = prosody_text.replace("now", "<STRONG>now")
            prosody_text = prosody_text.replace("today", "<STRONG>today")
            
        return prosody_text
        
    def _build_prosody_context(self,
                              analysis: 'ConversationAnalysis',
                              alignment: 'AlignmentAnalysis') -> Dict:
        """Build context for prosody application"""
        
        return {
            'sentiment': analysis.sentiment,
            'is_question': analysis.is_question,
            'is_addressing_concern': len(analysis.detected_concerns) > 0,
            'is_objection_response': analysis.question_type == 'objection',
            'is_presenting_benefit': any(word in str(alignment.detected_opportunities) 
                                       for word in ['benefit', 'save', 'improve']),
            'is_call_to_action': alignment.urgency_level == 'high',
            'urgency_level': alignment.urgency_level,
            'key_points': alignment.remaining_points[:2]  # Focus on next 2 points
        }
        
    def _select_voice(self,
                     analysis: 'ConversationAnalysis',
                     objective: 'PersuasionObjective') -> str:
        """Select appropriate voice based on context"""
        
        # Default voices
        professional_voices = ['en_man', 'en_woman']
        friendly_voices = ['belinda', 'mabel']
        authoritative_voices = ['chadwick', 'broom_salesman']
        
        # Select based on emotional state and context
        if analysis.emotional_state == 'skeptical':
            # Use authoritative voice for skeptical audience
            return authoritative_voices[0]
        elif analysis.emotional_state == 'interested':
            # Use friendly voice for interested audience
            return friendly_voices[0]
        else:
            # Default to professional
            return professional_voices[1]
            
    def _calculate_confidence(self, alignment: 'AlignmentAnalysis') -> float:
        """Calculate confidence score for response"""
        
        base_confidence = 0.7
        
        # Adjust based on alignment
        confidence = base_confidence + (alignment.alignment_score * 0.2)
        
        # Boost for opportunities
        if alignment.detected_opportunities:
            confidence += 0.1
            
        return min(confidence, 0.95)
        
    def _identify_tactics(self,
                         response_text: str,
                         alignment: 'AlignmentAnalysis') -> List[str]:
        """Identify persuasion tactics used in response"""
        
        tactics = []
        
        # Check for common persuasion patterns
        if any(word in response_text.lower() for word in ['proven', 'studies show', 'research']):
            tactics.append('appeal_to_authority')
            
        if any(word in response_text.lower() for word in ['save', 'benefit', 'gain']):
            tactics.append('highlight_benefits')
            
        if any(word in response_text.lower() for word in ['limited', 'now', 'today']):
            tactics.append('create_urgency')
            
        if any(word in response_text.lower() for word in ['understand', 'see your point']):
            tactics.append('empathy_building')
            
        if '%' in response_text or '$' in response_text:
            tactics.append('use_statistics')
            
        return tactics
        
    def _get_cache_key(self, transcript: str, analysis: 'ConversationAnalysis') -> str:
        """Generate cache key for response"""
        # Simple cache key based on question type and key topics
        return f"{analysis.question_type}:{':'.join(sorted(analysis.key_topics))}"
        
    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompt"""
        lines = []
        for exchange in history:
            if 'user' in exchange:
                lines.append(f"User: {exchange['user']}")
            if 'assistant' in exchange:
                lines.append(f"Assistant: {exchange['assistant']}")
        return '\n'.join(lines)


class SmoothStallGenerator:
    """Generates smooth stalling responses while processing"""
    
    STALL_PHRASES = {
        'thinking': [
            "That's an excellent question, let me think about that...",
            "I see what you're asking, give me a moment...",
            "That's definitely worth considering..."
        ],
        'clarifying': [
            "Just to make sure I understand correctly...",
            "That's an important point you're raising...",
            "Let me address that properly..."
        ],
        'acknowledging': [
            "I appreciate you bringing that up...",
            "That's a great observation...",
            "You've touched on something important..."
        ]
    }
    
    @staticmethod
    def get_stall_response(context_type: str = 'thinking') -> Tuple[str, str]:
        """Get appropriate stall response with prosody"""
        import random
        
        phrases = SmoothStallGenerator.STALL_PHRASES.get(
            context_type, 
            SmoothStallGenerator.STALL_PHRASES['thinking']
        )
        
        text = random.choice(phrases)
        
        # Add appropriate prosody
        prosody_text = f"<CALM> <FRIENDLY> {text} <PAUSE_LONG>"
        
        return text, prosody_text
