"""
Persuasion engine for objective alignment and strategy
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from openai import AsyncOpenAI
import json
from datetime import datetime


@dataclass
class PersuasionObjective:
    """Main persuasion objective configuration"""
    main_goal: str
    key_points: List[str]  # Points of Persuasion (POPs)
    audience_triggers: List[str]  # Words/concepts that resonate
    success_criteria: Optional[List[str]] = None
    objection_handlers: Optional[Dict[str, str]] = None


@dataclass
class AlignmentAnalysis:
    """Analysis of how current conversation aligns with objectives"""
    alignment_score: float  # 0-1 score
    addressed_points: List[str]
    remaining_points: List[str]
    detected_opportunities: List[str]
    suggested_pivot: Optional[str]
    urgency_level: str  # low, medium, high
    next_best_action: str


class PersuasionEngine:
    """Core engine for persuasion strategy and alignment"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.current_objective: Optional[PersuasionObjective] = None
        self.audience_profile: Optional[Dict] = None
        self.objective_progress: Dict[str, bool] = {}
        
    def set_objective(self, objective: PersuasionObjective):
        """Set the main persuasion objective"""
        self.current_objective = objective
        # Initialize progress tracking
        self.objective_progress = {point: False for point in objective.key_points}


    def set_audience_profile(self, profile: Dict):
        """Set audience profile for better targeting"""
        self.audience_profile = profile

    # valid languages: "english" and "french"
    async def check_alignment(self, 
                            transcript: str, 
                            analysis: 'ConversationAnalysis',
                            objective: PersuasionObjective,
                            language="english") -> AlignmentAnalysis:
        """Check how current conversation aligns with persuasion objectives"""
        
        prompts_dict = {"english": f"""Analyze persuasion alignment:

Persuasion Objective: {objective.main_goal}
Key Points (POPs): {json.dumps(objective.key_points)}
Audience Triggers: {json.dumps(objective.audience_triggers)}

Current Statement: "{transcript}"
Statement Analysis: 
- Sentiment: {analysis.sentiment}
- Is Question: {analysis.is_question}
- Concerns: {analysis.detected_concerns}
- Emotional State: {analysis.emotional_state}

Progress so far: {json.dumps(self.objective_progress)}

Provide strategic analysis in JSON:
{{
    "alignment_score": 0.0-1.0,
    "addressed_points": ["points covered in this exchange"],
    "remaining_points": ["points still to cover"],
    "detected_opportunities": ["opportunities to advance objective"],
    "suggested_pivot": "suggestion to steer conversation" or null,
    "urgency_level": "low/medium/high",
    "next_best_action": "specific action to take"
}}

Consider:
1. Are we making progress toward the objective?
2. What opportunities does this exchange present?
3. Should we pivot the conversation?
4. What's the most persuasive next step toward our goals?""",

"french": f"""Analyser l'alignement de la persuasion :

Objectif de persuasion : {objective.main_goal}
Points clés (POPs) : {json.dumps(objective.key_points)}
Déclencheurs pour l'audience : {json.dumps(objective.audience_triggers)}

Déclaration actuelle : "{transcript}"
Analyse de la déclaration : 
- Sentiment : {analysis.sentiment}
- Est-ce une question : {analysis.is_question}
- Préoccupations : {analysis.detected_concerns}
- État émotionnel : {analysis.emotional_state}

Progrès jusqu'à présent : {json.dumps(self.objective_progress)}

Fournir une analyse stratégique en JSON :
{{
    "score_d'alignement": 0.0-1.0,
    "points_abordés": ["points couverts dans cet échange"],
    "points_restants": ["points encore à couvrir"],
    "opportunités_détectées": ["opportunités pour avancer l'objectif"],
    "pivot_suggéré": "suggestion pour orienter la conversation" ou null,
    "niveau_d'urgence": "faible/moyenne/élevée",
    "prochaine_meilleure_action": "action spécifique à entreprendre"
}}

À considérer :
1. Avançons-nous vers l'objectif ?
2. Quelles opportunités cet échange présente-t-il ?
3. Devons-nous pivoter la conversation ?
4. Quelle est la prochaine étape la plus persuasive vers nos objectifs ?
"""}
        prompt = prompts_dict[language]
        try:
            response = await self.client.chat.completions.create(
                model=self.config.reasoning_model,
                messages=[
                    {"role": "system", "content": "You are an expert persuasion strategist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=512
            )
            
            alignment_data = json.loads(response.choices[0].message.content)
            
            # Update progress tracking
            for point in alignment_data.get("addressed_points", []):
                if point in self.objective_progress:
                    self.objective_progress[point] = True
                    
            return AlignmentAnalysis(
                alignment_score=alignment_data.get("alignment_score", 0.5),
                addressed_points=alignment_data.get("addressed_points", []),
                remaining_points=alignment_data.get("remaining_points", []),
                detected_opportunities=alignment_data.get("detected_opportunities", []),
                suggested_pivot=alignment_data.get("suggested_pivot"),
                urgency_level=alignment_data.get("urgency_level", "medium"),
                next_best_action=alignment_data.get("next_best_action", "continue")
            )
            
        except Exception as e:
            print(f"Alignment check error: {e}")
            return AlignmentAnalysis(
                alignment_score=0.5,
                addressed_points=[],
                remaining_points=objective.key_points,
                detected_opportunities=[],
                suggested_pivot=None,
                urgency_level="medium",
                next_best_action="continue conversation"
            )
            
    async def get_session_analytics(self, 
                                  history: List[Dict],
                                  objective: PersuasionObjective) -> Dict:
        """Generate analytics for the persuasion session"""
        
        # Calculate objective proximity over time
        proximity_timeline = []
        for i, exchange in enumerate(history):
            if 'alignment' in exchange:
                proximity_timeline.append({
                    'index': i,
                    'score': exchange['alignment'].alignment_score,
                    'timestamp': exchange.get('timestamp', i)
                })
                
        # Identify key moments
        key_moments = self._identify_key_moments(history)
        
        # Calculate success metrics
        addressed_count = sum(1 for v in self.objective_progress.values() if v)
        total_points = len(objective.key_points)
        completion_rate = addressed_count / total_points if total_points > 0 else 0
        
        return {
            'objective_completion': completion_rate,
            'proximity_timeline': proximity_timeline,
            'key_moments': key_moments,
            'addressed_points': [k for k, v in self.objective_progress.items() if v],
            'remaining_points': [k for k, v in self.objective_progress.items() if not v],
            'average_alignment': sum(p['score'] for p in proximity_timeline) / len(proximity_timeline) if proximity_timeline else 0,
            'recommendations': await self._generate_recommendations(history, objective)
        }
        
    def _identify_key_moments(self, history: List[Dict]) -> List[Dict]:
        """Identify key moments in the conversation"""
        key_moments = []
        
        for i, exchange in enumerate(history):
            if 'alignment' not in exchange:
                continue
                
            alignment = exchange['alignment']
            
            # High opportunity moments
            if alignment.detected_opportunities:
                key_moments.append({
                    'type': 'opportunity',
                    'index': i,
                    'description': f"Opportunity: {', '.join(alignment.detected_opportunities)}",
                    'impact': 'positive'
                })
                
            # Pivot moments
            if alignment.suggested_pivot:
                key_moments.append({
                    'type': 'pivot',
                    'index': i,
                    'description': f"Suggested pivot: {alignment.suggested_pivot}",
                    'impact': 'neutral'
                })
                
            # High urgency moments
            if alignment.urgency_level == 'high':
                key_moments.append({
                    'type': 'urgency',
                    'index': i,
                    'description': "High urgency moment requiring action",
                    'impact': 'critical'
                })
                
        return key_moments
        
    async def _generate_recommendations(self, 
                                      history: List[Dict],
                                      objective: PersuasionObjective) -> List[str]:
        """Generate recommendations for improving persuasion"""
        
        prompt = f"""Based on this persuasion session, provide recommendations:

Objective: {objective.main_goal}
Completion: {sum(1 for v in self.objective_progress.values() if v)}/{len(objective.key_points)} points addressed

Key moments identified: {len(self._identify_key_moments(history))}

Provide 3-5 specific recommendations for improving persuasion effectiveness in future conversations."""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.reasoning_model,
                messages=[
                    {"role": "system", "content": "You analyze persuasion sessions and provide improvement recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=256
            )
            
            # Extract recommendations from response
            recommendations = response.choices[0].message.content.strip().split('\n')
            return [r.strip() for r in recommendations if r.strip()]
            
        except Exception as e:
            print(f"Recommendation generation error: {e}")
            return ["Review unaddressed points", "Focus on audience triggers", "Practice objection handling"]


class AudienceProfiler:
    """Analyzes and adapts to audience characteristics"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
    async def enhance_objective(self,
                               objective: PersuasionObjective,
                               audience_profile: Dict) -> PersuasionObjective:
        """Enhance persuasion objective based on audience profile"""
        
        prompt = f"""Enhance this persuasion strategy for the specific audience:

Original Objective: {objective.main_goal}
Original Points: {json.dumps(objective.key_points)}

Audience Profile:
{json.dumps(audience_profile, indent=2)}

Provide enhanced strategy in JSON:
{{
    "refined_goal": "more targeted goal statement",
    "enhanced_points": ["audience-specific persuasion points"],
    "recommended_triggers": ["words/concepts that resonate with this audience"],
    "avoid_topics": ["topics to avoid with this audience"],
    "communication_style": "recommended style"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.config.reasoning_model,
                messages=[
                    {"role": "system", "content": "You are an expert at audience analysis and persuasion customization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            
            enhancement_data = json.loads(response.choices[0].message.content)
            
            # Create enhanced objective
            enhanced_objective = PersuasionObjective(
                main_goal=enhancement_data.get("refined_goal", objective.main_goal),
                key_points=enhancement_data.get("enhanced_points", objective.key_points),
                audience_triggers=enhancement_data.get("recommended_triggers", objective.audience_triggers),
                objection_handlers={
                    "default": "I understand your concern. Let me address that..."
                }
            )
            
            return enhanced_objective
            
        except Exception as e:
            print(f"Audience profiling error: {e}")
            return objective  # Return original if enhancement fails
