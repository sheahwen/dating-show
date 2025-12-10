"""Conversation flow control for the dating show."""

import random
from typing import List, Optional
from autogen_agentchat.messages import TextMessage

from stage1.agents.director import DirectorAgent
from stage1.agents.participant import ParticipantAgent


class ConversationController:
    """
    Handles conversation flow logic including speaker selection and director interventions.

    Speaker Selection Algorithm:
    1. Gets interest scores (0.0-1.0) from all participants based on recent conversation
    2. Applies fairness adjustment: participants who have spoken less get a small boost (up to 0.1)
    3. Filters participants by interest thresholds:
       - Primary: interest >= 0.4 (high interest)
       - Fallback: interest >= 0.2 (moderate interest)
       - Final fallback: random selection if no one is interested
    4. Uses weighted random selection based on adjusted scores

    Director Intervention Logic:
    - Combines director's conversation analysis with random probability
    - Director analyzes conversation flow and decides if intervention is needed
    - Random chance provides additional unpredictability for dramatic effect
    """

    def __init__(
        self,
        participants: List[ParticipantAgent],
        director: DirectorAgent,
        director_intervention_probability: float = 0.05,
    ):
        self.participants = participants
        self.director = director
        self.director_intervention_probability = director_intervention_probability
        self.participant_speak_counts = {p.name: 0 for p in participants}

    def increment_speak_count(self, participant_name: str) -> None:
        """Increment the speak count for a participant."""
        if participant_name in self.participant_speak_counts:
            self.participant_speak_counts[participant_name] += 1

    async def check_director_intervention(
        self, recent_messages: List[TextMessage]
    ) -> bool:
        """
        Check if the director should intervene based on conversation analysis and probability.

        Returns:
            True if director should intervene, False otherwise
        """
        try:
            director_analysis = await self.director.should_intervene(recent_messages)

            random_chance = random.random() < self.director_intervention_probability

            should_intervene = director_analysis or (
                random_chance and len(recent_messages) > 3
            )

            return should_intervene
        except Exception as e:
            print(f"Error checking director intervention: {e}")
            return random.random() < self.director_intervention_probability

    async def select_next_speaker(
        self, recent_messages: List[TextMessage]
    ) -> Optional[ParticipantAgent]:
        """
        Select the next speaker based on interest scores and organic conversation flow.

        Returns:
            The participant who should speak next, or None if no one wants to speak
        """
        if len(self.participants) == 0:
            return None

        try:
            interest_scores = {}
            for participant in self.participants:
                score = await participant.get_speaking_interest(recent_messages)
                interest_scores[participant.name] = score
                print(f"  {participant.name} interest: {score:.2f}")

            adjusted_scores = {}
            total_speaks = sum(self.participant_speak_counts.values())

            for participant in self.participants:
                base_score = interest_scores[participant.name]
                speak_count = self.participant_speak_counts[participant.name]

                if total_speaks > 0:
                    fairness_boost = 0.1 * (
                        1 - speak_count / max(1, total_speaks / len(self.participants))
                    )
                else:
                    fairness_boost = 0.0

                adjusted_scores[participant.name] = base_score + fairness_boost

            interested_participants = [
                p for p in self.participants if adjusted_scores[p.name] > 0.4
            ]

            if not interested_participants:
                interested_participants = [
                    p for p in self.participants if adjusted_scores[p.name] > 0.2
                ]

            if not interested_participants:
                return random.choice(self.participants)

            adjusted_weights = [
                adjusted_scores[p.name] for p in interested_participants
            ]
            selected = random.choices(
                interested_participants, weights=adjusted_weights
            )[0]

            return selected

        except Exception as e:
            print(f"Error selecting next speaker: {e}")
            return random.choice(self.participants)
