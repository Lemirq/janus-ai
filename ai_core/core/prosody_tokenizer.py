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
    Prosody tokens have IDs >= 5,000,000 to avoid collisions with other token IDs.
    """
    
    # Define prosody tokens with IDs >= 5,000,000
    PROSODY_TOKENS = {
        # Emphasis
        "<emph>": ProsodyToken("emphasis", 5000000, "Emphasize next word", "<emph>"),

        # Pause control
        "<pause_short>": ProsodyToken("pause_short", 5000001, "Short pause", "<pause_short>"),
        "<pause_long>": ProsodyToken("pause_long", 5000002, "Long pause", "<pause_long>"),
        
        # Pitch control
        "<pitch_high>": ProsodyToken("pitch_high", 5000003, "Higher pitch", "<pitch_high>"),
        "<pitch_low>": ProsodyToken("pitch_low", 5000004, "Lower pitch", "<pitch_low>"),
        "<pitch_rising>": ProsodyToken("pitch_rising", 5000005, "Rising intonation", "<pitch_rising>"),
        "<pitch_falling>": ProsodyToken("pitch_falling", 5000006, "Falling intonation", "<pitch_falling>"),
    }
    
    def __init__(self, base_model_name: str = "gpt2"):
        """Initialize with base tokenizer"""
        # Use GPT-2 tokenizer instead of gated Llama model
        # For production, you can use the actual model tokenizer once authenticated
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer {base_model_name}, using simple fallback")
            self.base_tokenizer = None
        
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
                if self.base_tokenizer:
                    # Use base tokenizer for regular text
                    text_tokens = self.base_tokenizer.encode(part, add_special_tokens=False)
                    encoded_tokens.extend(text_tokens)
                else:
                    # Fallback: simple word-based tokenization
                    # This is just for demo - production would use proper tokenizer
                    encoded_tokens.extend([hash(word) % 100000 for word in part.split()])
                
        return encoded_tokens
        
    def decode_with_prosody(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text with prosody markers.
        """
        result = []
        regular_tokens = []
        
        for token_id in token_ids:
            if token_id >= 5000000 and token_id in self.id_to_token:
                # First decode any accumulated regular tokens
                if regular_tokens:
                    if self.base_tokenizer:
                        result.append(self.base_tokenizer.decode(regular_tokens))
                    else:
                        result.append(" [tokens] ")
                    regular_tokens = []
                # Add prosody token
                result.append(self.id_to_token[token_id])
            else:
                # Accumulate regular tokens
                regular_tokens.append(token_id)
                
        # Decode any remaining regular tokens
        if regular_tokens:
            if self.base_tokenizer:
                result.append(self.base_tokenizer.decode(regular_tokens))
            else:
                result.append(" [tokens] ")
            
        return ''.join(result)
        
    def apply_prosody_rules(self, text: str, context: Dict = None) -> str:
        """
        Apply intelligent prosody based on content and context.
        """
        if context is None:
            context = {}
            
        # Start with original text
        prosody_text = text
        
        # Apply emphasis to key persuasion points
        if 'key_points' in context:
            for point in context['key_points']:
                if point.lower() in text.lower():
                    # Find and emphasize the key point
                    pattern = re.compile(re.escape(point), re.IGNORECASE)
                    prosody_text = pattern.sub(f"<emph>{point}", prosody_text, count=1)
            
        # Add strategic pauses before important points
        prosody_text = re.sub(r'(\. )([A-Z])', r'\1<pause_short> \2', prosody_text)
        
        # Before numbers/statistics (persuasive emphasis)
        prosody_text = re.sub(r'(\s)(\d+%|\$\d+|\d+ percent)', r'\1<emph>\2', prosody_text)
        
        # Question intonation
        if '?' in text:
            prosody_text = re.sub(r'([^?]+)(\?)', r'\1<pitch_rising>\2', prosody_text)
                
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
                # Reset prosody after applying (all are one-time effects now)
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
        
        # Handling objections - pause before response
        if persuasion_context.get('is_objection_response', False):
            prosody_text = f"<pause_short> {prosody_text}"
            # Emphasize understanding
            if "understand" in prosody_text.lower():
                prosody_text = re.sub(r'\bunderstand\b', '<emph>understand', prosody_text, flags=re.IGNORECASE)
            
        # Call to action - emphasize action words
        if persuasion_context.get('is_call_to_action', False):
            action_words = ['now', 'today', 'immediately', 'act', 'decide', 'choose']
            for word in action_words:
                if word in prosody_text.lower():
                    pattern = re.compile(f'\\b{word}\\b', re.IGNORECASE)
                    prosody_text = pattern.sub(f'<emph>{word}', prosody_text, count=1)
                    
        # Statistics and evidence - emphasize numbers
        stat_pattern = r'(\d+%|\$[\d,]+|\d+ out of \d+)'
        prosody_text = re.sub(stat_pattern, r'<emph>\1', prosody_text)
        
        # Benefits - emphasize with rising pitch
        benefit_keywords = ['benefit', 'advantage', 'improve', 'increase', 'save', 'gain']
        for keyword in benefit_keywords:
            if keyword in prosody_text.lower():
                pattern = re.compile(f'(\\b{keyword})', re.IGNORECASE)
                prosody_text = pattern.sub(r'<pitch_rising>\1', prosody_text, count=1)
                
        return prosody_text
