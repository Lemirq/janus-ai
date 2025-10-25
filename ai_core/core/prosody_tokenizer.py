"""
Custom prosody tokenizer for Higgs model
Handles prosody tokens with IDs >= 128,000 for audio stream injection
"""

from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class ProsodyToken:
    """Represents a prosody control token"""
    name: str
    token_id: int
    description: str
    symbol: str  # Text representation in prompt


class ProsodyTokenizer:
    """
    Custom tokenizer that handles prosody tokens for Higgs model.
    Prosody tokens have IDs >= 128,000 to be passed to audio stream.
    """
    
    # Define prosody tokens with IDs >= 128,000
    PROSODY_TOKENS = {
        # Emphasis and stress
        "<EMPH>": ProsodyToken("emphasis", 128000, "Emphasize next word", "<EMPH>"),
        "<STRONG>": ProsodyToken("strong", 128001, "Strong emphasis", "<STRONG>"),
        "<SOFT>": ProsodyToken("soft", 128002, "Soft delivery", "<SOFT>"),
        
        # Pace control
        "<SLOW>": ProsodyToken("slow", 128003, "Slow down pace", "<SLOW>"),
        "<FAST>": ProsodyToken("fast", 128004, "Speed up pace", "<FAST>"),
        "<PAUSE_SHORT>": ProsodyToken("pause_short", 128005, "Short pause", "<PAUSE_SHORT>"),
        "<PAUSE_LONG>": ProsodyToken("pause_long", 128006, "Long pause", "<PAUSE_LONG>"),
        
        # Pitch control
        "<PITCH_HIGH>": ProsodyToken("pitch_high", 128007, "Higher pitch", "<PITCH_HIGH>"),
        "<PITCH_LOW>": ProsodyToken("pitch_low", 128008, "Lower pitch", "<PITCH_LOW>"),
        "<PITCH_RISE>": ProsodyToken("pitch_rise", 128009, "Rising intonation", "<PITCH_RISE>"),
        "<PITCH_FALL>": ProsodyToken("pitch_fall", 128010, "Falling intonation", "<PITCH_FALL>"),
        
        # Emotion and tone
        "<CONFIDENT>": ProsodyToken("confident", 128011, "Confident tone", "<CONFIDENT>"),
        "<FRIENDLY>": ProsodyToken("friendly", 128012, "Friendly tone", "<FRIENDLY>"),
        "<SERIOUS>": ProsodyToken("serious", 128013, "Serious tone", "<SERIOUS>"),
        "<EXCITED>": ProsodyToken("excited", 128014, "Excited tone", "<EXCITED>"),
        "<CALM>": ProsodyToken("calm", 128015, "Calm tone", "<CALM>"),
        
        # Reset
        "<RESET>": ProsodyToken("reset", 128016, "Reset to normal", "<RESET>")
    }
    
    def __init__(self, base_model_name: str = "meta-llama/Llama-3.2-3B"):
        """Initialize with base tokenizer"""
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.prosody_pattern = re.compile(r'(<[A-Z_]+>)')
        
        # Create reverse lookup
        self.token_to_id = {token.symbol: token.token_id for token in self.PROSODY_TOKENS.values()}
        self.id_to_token = {token.token_id: token.symbol for token in self.PROSODY_TOKENS.values()}
        
    def encode_with_prosody(self, text: str) -> List[int]:
        """
        Encode text with prosody tokens.
        Prosody tokens are converted to their special IDs (>= 128000).
        """
        # Split text by prosody tokens
        parts = self.prosody_pattern.split(text)
        
        encoded_tokens = []
        
        for part in parts:
            if part in self.token_to_id:
                # This is a prosody token
                encoded_tokens.append(self.token_to_id[part])
            elif part:  # Non-empty text
                # Use base tokenizer for regular text
                text_tokens = self.base_tokenizer.encode(part, add_special_tokens=False)
                encoded_tokens.extend(text_tokens)
                
        return encoded_tokens
        
    def decode_with_prosody(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text with prosody markers.
        """
        result = []
        regular_tokens = []
        
        for token_id in token_ids:
            if token_id >= 128000 and token_id in self.id_to_token:
                # First decode any accumulated regular tokens
                if regular_tokens:
                    result.append(self.base_tokenizer.decode(regular_tokens))
                    regular_tokens = []
                # Add prosody token
                result.append(self.id_to_token[token_id])
            else:
                # Accumulate regular tokens
                regular_tokens.append(token_id)
                
        # Decode any remaining regular tokens
        if regular_tokens:
            result.append(self.base_tokenizer.decode(regular_tokens))
            
        return ''.join(result)
        
    def apply_prosody_rules(self, text: str, sentiment: str, context: Dict) -> str:
        """
        Apply intelligent prosody based on content and context.
        """
        # Start with original text
        prosody_text = text
        
        # Apply emphasis to key persuasion points
        if 'key_points' in context:
            for point in context['key_points']:
                if point.lower() in text.lower():
                    # Find and emphasize the key point
                    pattern = re.compile(re.escape(point), re.IGNORECASE)
                    prosody_text = pattern.sub(f"<EMPH>{point}", prosody_text, count=1)
                    
        # Apply tone based on sentiment
        tone_mapping = {
            'positive': '<FRIENDLY>',
            'confident': '<CONFIDENT>',
            'serious': '<SERIOUS>',
            'excited': '<EXCITED>',
            'calm': '<CALM>'
        }
        
        if sentiment in tone_mapping:
            prosody_text = f"{tone_mapping[sentiment]} {prosody_text}"
            
        # Add strategic pauses
        # Before important points
        prosody_text = re.sub(r'(\. )([A-Z])', r'\1<PAUSE_SHORT> \2', prosody_text)
        
        # Before numbers/statistics (persuasive emphasis)
        prosody_text = re.sub(r'(\s)(\d+%|\$\d+|\d+ percent)', r'\1<PAUSE_SHORT> <EMPH>\2', prosody_text)
        
        # Question intonation
        if '?' in text:
            prosody_text = re.sub(r'(\?)', r'<PITCH_RISE>\1', prosody_text)
            
        # Confidence boosters
        confidence_phrases = ['guarantee', 'proven', 'definitely', 'certainly', 'absolutely']
        for phrase in confidence_phrases:
            if phrase in text.lower():
                pattern = re.compile(f'\\b{phrase}\\b', re.IGNORECASE)
                prosody_text = pattern.sub(f'<STRONG>{phrase}', prosody_text)
                
        return prosody_text
        
    def extract_prosody_sequence(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """
        Extract sequence of (text, prosody) tuples for processing.
        """
        parts = self.prosody_pattern.split(text)
        
        sequence = []
        current_prosody = None
        
        for part in parts:
            if part in self.token_to_id:
                # This is a prosody token
                current_prosody = part
            elif part:  # Non-empty text
                sequence.append((part, current_prosody))
                # Reset prosody after applying (unless it's a tone setting)
                if current_prosody and current_prosody not in ['<FRIENDLY>', '<CONFIDENT>', '<SERIOUS>', '<CALM>']:
                    current_prosody = None
                    
        return sequence
        
    def get_prosody_info(self) -> Dict[str, Dict]:
        """Get information about available prosody tokens"""
        return {
            token.symbol: {
                'name': token.name,
                'id': token.token_id,
                'description': token.description
            }
            for token in self.PROSODY_TOKENS.values()
        }


class ProsodyStrategy:
    """Strategic application of prosody for persuasion"""
    
    @staticmethod
    def apply_persuasion_prosody(text: str, persuasion_context: Dict) -> str:
        """Apply prosody specifically for persuasive effect"""
        
        prosody_text = text
        
        # Opening - establish authority and friendliness
        if persuasion_context.get('is_opening', False):
            prosody_text = f"<FRIENDLY> <CONFIDENT> {prosody_text}"
            
        # Handling objections - calm and confident
        if persuasion_context.get('is_objection_response', False):
            prosody_text = f"<CALM> <PAUSE_SHORT> {prosody_text}"
            # Emphasize understanding
            prosody_text = prosody_text.replace("understand", "<EMPH>understand")
            
        # Call to action - energy and urgency
        if persuasion_context.get('is_call_to_action', False):
            prosody_text = f"<EXCITED> {prosody_text}"
            # Emphasize action words
            action_words = ['now', 'today', 'immediately', 'act', 'decide', 'choose']
            for word in action_words:
                if word in prosody_text.lower():
                    pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
                    prosody_text = pattern.sub(f'<STRONG>{word}', prosody_text)
                    
        # Statistics and evidence - slow down and emphasize
        stat_pattern = r'(\d+%|\$[\d,]+|\d+ out of \d+)'
        prosody_text = re.sub(stat_pattern, r'<SLOW> <EMPH>\1<RESET>', prosody_text)
        
        # Benefits - positive and uplifting
        benefit_keywords = ['benefit', 'advantage', 'improve', 'increase', 'save', 'gain']
        for keyword in benefit_keywords:
            if keyword in prosody_text.lower():
                # Add positive tone before benefits
                pattern = re.compile(f'(\\b{keyword})', re.IGNORECASE)
                prosody_text = pattern.sub(r'<PITCH_RISE>\1', prosody_text)
                
        return prosody_text
