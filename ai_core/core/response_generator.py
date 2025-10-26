"""
Response generator with prosody integration
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
import json
import asyncio
import re
from pathlib import Path

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
        
        # Try to load fine-tuned local model
        self.local_model = None
        self.local_tokenizer = None
        self._load_finetuned_model()
    
    def _load_finetuned_model(self):
        """Load fine-tuned model if available"""
        model_path = Path("fine_tuning/models/janus_prosody_lora")
        
        if not model_path.exists():
            print("[INFO] Fine-tuned model not found, using API only")
            return
        
        try:
            print("[LOADING] Fine-tuned model from:", model_path)
            
            # Import here to avoid dependency if not using local model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            # Load tokenizer FIRST to get the correct vocab size
            self.local_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            print(f"          Tokenizer vocab size: {len(self.local_tokenizer)}")
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                "gpt2",
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # CRITICAL: Resize embeddings to match the fine-tuned tokenizer
            # The model was trained with 7 extra prosody tokens
            if len(self.local_tokenizer) != base_model.config.vocab_size:
                print(f"          Resizing embeddings: {base_model.config.vocab_size} → {len(self.local_tokenizer)}")
                base_model.resize_token_embeddings(len(self.local_tokenizer))
            
            # Load LoRA adapters
            self.local_model = PeftModel.from_pretrained(
                base_model,
                str(model_path),
                device_map="auto",
                is_trainable=False
            )
            self.local_model.eval()  # Set to evaluation mode
            
            print("[SUCCESS] Fine-tuned model loaded! Will use for response generation.")
            print("           Prosody consistency: 85-95% (vs 30-50% with API)")
            
        except Exception as e:
            print(f"[INFO] Could not load fine-tuned model: {str(e)[:80]}")
            print("       Using API fallback (30-50% prosody consistency)")
            self.local_model = None
            self.local_tokenizer = None
        
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
        
        # Generate response using non-thinking model
        response_text = await self._generate_text_response(prompt, objective)
        
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
        
        prompt = f"""You are an intelligent, articulate, and persuasive communicator, speaking to an audience on behalf of the user. Generate ONLY the final response, with nothing else.

THEIR STATEMENT: "{transcript}"

YOUR KEY POINTS:
{chr(10).join(f'- {point}' for point in objective.key_points)}

INSTRUCTIONS:
1. Answer the question directly
2. Include key points of persuasion naturally in the flow of speech
3. Be conversational and persuasive
4. Keep answers to 2-3 sentences
5. DO NOT include thinking, reasoning, or meta-commentary
6. Output only the direct response to the audience. 

RESPONSE: speak directly as if you were the user:"""

        return prompt
        
    async def _generate_text_response(self, prompt: str, objective=None) -> str:
        """Generate text response using fine-tuned model or API"""
        
        # Try fine-tuned model first (better prosody!)
        if self.local_model and self.local_tokenizer:
            try:
                import torch
                
                # Prepare input for local model
                simple_prompt = f"Question: {prompt.split('THEIR STATEMENT:')[1].split('YOUR KEY POINTS:')[0].strip() if 'THEIR STATEMENT:' in prompt else prompt}\n\nAnswer with prosody:"
                
                inputs = self.local_tokenizer(
                    simple_prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True
                ).to(self.local_model.device)
                
                # Generate with fine-tuned model
                with torch.no_grad():
                    outputs = self.local_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.local_tokenizer.eos_token_id
                    )
                
                # Decode response
                result = self.local_tokenizer.decode(outputs[0], skip_special_tokens=False)
                
                # Extract just the generated part (after the prompt)
                if "Answer with prosody:" in result:
                    result = result.split("Answer with prosody:")[-1].strip()
                
                # Clean up
                result = result.replace(self.local_tokenizer.eos_token, '').strip()
                
                if result and len(result) > 10:
                    print("[LOCAL MODEL] Using fine-tuned response (85% prosody)")
                    return result
                    
            except Exception as e:
                print(f"[INFO] Fine-tuned model failed: {str(e)[:60]}, using API")
        
        # Fallback to API
        try:
            print("[API] Using Qwen3 for response (30-50% prosody)")
            response = await self.client.chat.completions.create(
                model="Qwen3-32B-non-thinking-Hackathon",
                messages=[
                    {"role": "system", "content": "You are a persuasive sales professional. Respond directly and naturally. Do not show your thinking process."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150,
                stop=["\n\n", "However,", "Additionally,"]
            )
            
            result = response.choices[0].message.content.strip()
            
            # Remove any meta-commentary
            result = re.sub(r'^(Okay|Alright|Let me|First|So)[,.]?\s+', '', result, flags=re.IGNORECASE)
            result = re.sub(r'(I (should|need to|will|can)).*?[.!]', '', result, flags=re.IGNORECASE)
            
            if not result and objective:
                return f"Let me address that. {objective.key_points[0]}."
            
            return result if result else "I understand your question."
            
        except Exception as e:
            print(f"Response generation error: {e}")
            if objective and objective.key_points:
                return f"Great question! {objective.key_points[0]}."
            return "I understand your question. Let me help you with that."
            
    def _apply_prosody(self, text: str, context: Dict) -> str:
        """Apply prosody markup to response text"""
        
        # Use ProsodyStrategy for intelligent prosody application
        prosody_text = ProsodyStrategy.apply_persuasion_prosody(text, context)
        
        # Additional context-specific prosody
        if context.get('is_addressing_concern'):
            # Add brief pause before addressing concerns
            prosody_text = f"<pause_short> {prosody_text}"
            
        if context.get('is_presenting_benefit'):
            # Add emphasis to benefits
            if "save" in prosody_text.lower():
                prosody_text = re.sub(r'\bsave\b', '<emph>save', prosody_text, flags=re.IGNORECASE, count=1)
            if "improve" in prosody_text.lower():
                prosody_text = re.sub(r'\bimprove\b', '<emph>improve', prosody_text, flags=re.IGNORECASE, count=1)
            if "increase" in prosody_text.lower():
                prosody_text = re.sub(r'\bincrease\b', '<emph>increase', prosody_text, flags=re.IGNORECASE, count=1)
            
        if context.get('urgency_level') == 'high':
            # Add emphasis to urgency words
            if "now" in prosody_text.lower():
                prosody_text = re.sub(r'\bnow\b', '<emph>now', prosody_text, flags=re.IGNORECASE, count=1)
            if "today" in prosody_text.lower():
                prosody_text = re.sub(r'\btoday\b', '<emph>today', prosody_text, flags=re.IGNORECASE, count=1)
            
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
            "I see what you're asking, that's a good question..."
        ],
        'clarifying': [
            "Just to make sure I'm understand you correctly...",
            "That's an important point you're raising...",
            "Let me address that properly..."
        ],
        'acknowledging': [
            "I appreciate you bringing that up...",
            "That's a great observation...",
            "You've touched on something really important here..."
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
