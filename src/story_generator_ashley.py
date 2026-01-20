#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import random
from typing import Dict, Any, Optional
from llamaindex_interface import ChatWithRAG
from llm_prompts import ConversationPrompt


class StoryGenerator:
    """
    Generates therapeutic stories using the LLM system
    """
    STORY_PROMPT_TEMPLATE_1 = """Write a short therapeutic story for a {age}-year-old child named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones with a three act structure that teaches an important lesson.

    Use the following story template: A protagonist named {child_name} starts a journey. Along the way, {child_name} encounters one or two obstacles while trying to reach a reward. {child_name} meets supportive characters who help {child_name}. With their help, {child_name} achieves {child_name}'s goal and finds the reward.

    Integrate the following speech and language therapy goals naturally into the story:
    1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.

    The story should be: Short (about 300–500 words). Written in simple and friendly language.

    Keep in mind the following goals for {child_name} while generating the story: {goals}"""

    SCHOOL_TEMPLATE_3 = """Write a short therapeutic story for a {age}-year-old child named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones with an emphasis on learning how to do an everyday task.

        Use the following story template: A protagonist named {child_name} is going to their first day of school. {child_name} worries about making friends and doesn't want to go but decides to be brave and overcome their fear. {child_name} meets supportive friends who help {child_name} feel excited about school. With their help, {child_name} overcomes their worries and enjoys school and learning.

        Integrate the following speech and language therapy goals naturally into the story:
        1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.

        The story should be: Short (about 300–500 words). Written in simple and friendly language.

        Keep in mind the following goals for {child_name} while generating the story: {goals}"""

    STORY_PROMPT_TEMPLATE_3 = """Write a short therapeutic story for a {age}-year-old child named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones with an emphasis on learning how to do an everyday task.

            Use the following story template: A protagonist named {child_name} is promenading in the forest. {child_name} finds a lost small animal who has been separated from their family. {child_name} decides to help them reunite with their family, together they brave challenges throughout the forest and learn the importance of friendhsip. At the end {child_name} reunites their friend with their family and feels gratified knowing they were able to help. 

            Integrate the following speech and language therapy goals naturally into the story:
            1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.

            The story should be: Short (about 300–500 words). Written in simple and friendly language.

            Keep in mind the following goals for {child_name} while generating the story: {goals}"""

    SCHOOL_TEMPLATE_1 = """Write a children’s story titled The Lost Lunchbox for a {age}-year-old. The story should center on the theme of friendship, curiosity, and problem-solving, and take place in an elementary school during lunchtime. 

                            You may also include side characters like a kind teacher, a helpful classmate, or even a playful classroom pet. 

                            Begin by introducing the characters, the school setting, and the main problem—{child_name} discovers his brand-new lunchbox is missing. In the middle, show how friends can work together to solve the mystery, face small challenges, and learn the value of teamwork and empathy. End with a warm and satisfying resolution where the lunchbox is found and {child_name} realizes that helping others and staying kind always pays off. Use simple, expressive language, short sentences, and lively dialogue appropriate for young readers. 

                            Keep the tone friendly, imaginative, and lighthearted, about 400–700 words long. Add playful touches of humor, sound effects, or small moments of wonder to make the story engaging and vivid for children.
                            Keep in mind the following goals for {child_name} while generating the story: {goals}
                            """

    SCHOOL_TEMPLATE_2 = """Tell a gentle, encouraging story about a young child named {child_name} who is going to school for the very first time. {child_name} feels a mix of excitement and worry while getting ready in the morning — {child_name}  doesn’t know what the day will be like or if they'll make any friends. {child_name}'s stomach feels fluttery, and everything seems big and new.

When {child_name} arrives at school, {child_name} discovers a colorful classroom filled with books, toys, and smiling faces. {child_name}'s teacher greets them kindly and helps them find a seat. At first, {child_name} feels shy, but soon another child asks if to play. Together, they build a tower, draw pictures, and laugh. Slowly, {child_name}'s nervousness begins to fade.

By the end of the day, {child_name} realizes that school is a place where one can have fun, learn new things, and be themselves. {child_name} feels proud for being brave and is excited to come back tomorrow.

Keep in mind the following goals for {child_name} while generating the story: {goals}
"""

    NATURE_TEMPLATE_1 = """Write a lyrical, nature-based children’s story titled “{child_name} and the Whispering Wind.” The story centers on {child_name}, a curious young child who loves to listen to the wind in the trees. When a sudden storm arrives, they learn how nature can be both strong and gentle — and that calm always returns after chaos.

        Follow a three-part story structure:
        Beginning: Introduce {child_name}’s fascination with the wind and their sense of wonder.
        Middle: Describe the storm through vivid sensory imagery — whistling leaves, bending branches, soft rain.
        End: Show the stillness after the storm, as {child_name} learns courage, patience, and appreciation for balance in nature.

        Integrate learning goals about emotional regulation, observation, and respect for natural forces. Focus on rhythm and sensory language to build vocabulary — whooshing, rustling, shimmering. The tone should be poetic, soothing, and reflective.

        Keep in mind the following goals for {child_name} while generating the story: {goals}
"""

    NATURE_TEMPLATE_2 = """Write a peaceful children’s story titled River’s Secret Song. The story follows {child_name}, who loves to play by the river but can’t understand why it sounds different each day. Guided by a wise frog, {child_name} discovers that the river changes its tune with the weather, the seasons, and the creatures around it — just like people do.

    Follow a three-part story structure:
    Beginning: Introduce {child_name}’s curiosity and their visits to the river.
    Middle: Show their exploration and conversations with the frog as he listens carefully to nature’s “music.”
    End: End with {child_name} recognizing the beauty of change and harmony in the world around them.

    Highlight learning goals on mindfulness, listening skills, curiosity, and appreciation of the environment. Use gentle, musical language — rippling, humming, splashing — to support auditory awareness and descriptive understanding. The tone should be calm, rhythmic, and full of quiet wonder.

    Keep in mind the following goals for {child_name} while generating the story: {goals}
    """

    SCHOOL_STORY_PROMPT_TEMPLATES = {"template_3": SCHOOL_TEMPLATE_3, "template_1": SCHOOL_TEMPLATE_1,
                                     "template_2": SCHOOL_TEMPLATE_2}

    NATURE_STORY_PROMPT_TEMPLATES = {"template_1": NATURE_TEMPLATE_1, "template_2": NATURE_TEMPLATE_2,
                                     }

    THEME_TEMPLATES = {"SCHOOL_STORY_PROMPT_TEMPLATES": SCHOOL_STORY_PROMPT_TEMPLATES, "NATURE_STORY_PROMPT_TEMPLATES": NATURE_STORY_PROMPT_TEMPLATES}

    STORY_PROMPT_TEMPLATE = """Write a short therapeutic story for a {age}-year-old child named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones with a three act structure that teaches an important lesson.

    Use the following story template: A protagonist named {child_name} starts a journey. Along the way, {child_name} encounters one or two obstacles while trying to reach a reward. {child_name} meets supportive characters who help {child_name}. With their help, {child_name} achieves {child_name}'s goal and finds the reward.

    Integrate the following speech and language therapy goals naturally into the story:
    1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.

    The story should be: Short (about 300–500 words). Written in simple and friendly language."""

    def __init__(self, llm_model: str = "llama3.1", disable_rag: bool = True):
        """
        Initialize the story generator

        Args:
            llm_model: The LLM model to use for story generation
            disable_rag: Whether to disable RAG for story generation (recommended for creative tasks)
        """
        self.llm_model = llm_model
        self.disable_rag = disable_rag
        self.chat_engine = None
        self._initialize_chat_engine()

    def _initialize_chat_engine(self):
        """Initialize the chat engine for story generation"""
        # Use a specialized system prompt for story generation

        story_system_prompt = """You are a creative storyteller specializing in therapeutic stories for children with speech delays. 
        Your stories should be:
        - Engaging and age-appropriate
        - Supportive of language development
        - Simple in structure but rich in descriptive language
        - Encouraging and positive in tone
        - Around 300-500 words in length
        - Only output the story text and nothing else

        Focus on creating stories that help children learn descriptive words, overcoming challenges, practicing collaboration, and the importance of friendships and relationships

        """

        self.chat_engine = ChatWithRAG(
            model=self.llm_model,
            system_role=story_system_prompt,
            disable_rag=self.disable_rag,
            max_tokens=1000  # Allow for longer story generation
        )

    def generate_story(self, child_name: str, age: int, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a therapeutic story for a child

        Args:
            child_name: Name of the child for the story
            age: Age of the child
            custom_prompt: Optional custom prompt to override the default template

        Returns:
            Dictionary containing the generated story and metadata
        """
        with open("src/user.json", "r") as data_file:
            data = json.load(data_file)
        goals = data[child_name]["learning_goals"]
        try:
            if not self.chat_engine:
                return {
                    "success": False,
                    "error": "Chat engine not initialized",
                    "story": None,
                    "metadata": None
                }

            # Use custom prompt if provided, otherwise use template
            if custom_prompt:
                prompt = custom_prompt
            else:
                rand_theme = random.randint(0, len(self.THEME_TEMPLATES)-1)
                theme = self.THEME_TEMPLATES[rand_theme]
                rand_story = random.randint(0, len(self.theme) - 1)
                prompt = self.STORY_PROMPT_TEMPLATE[rand_story].format(
                    child_name=child_name,
                    age=age,
                    goals=goals
                )

            # Generate the story using the chat engine
            response = self.chat_engine.get_response(prompt)
            story_text = response.message.content if hasattr(response, 'message') else str(response)

            # Create metadata for the story
            story_metadata = {
                "child_name": child_name,
                "age": age,
                "word_count": len(story_text.split()),
                "generated_at": str(response.created_at) if hasattr(response, 'created_at') else None,
                "model": self.llm_model
            }

            return {
                "success": True,
                "story": story_text,
                "metadata": story_metadata
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "story": None,
                "metadata": None
            }

    def generate_story_stream(self, child_name: str, age: int, custom_prompt: Optional[str] = None):
        """
        Generate a therapeutic story with streaming response

        Args:
            child_name: Name of the child for the story
            age: Age of the child
            custom_prompt: Optional custom prompt to override the default template

        Yields:
            Story text chunks as they are generated
        """
        try:
            if not self.chat_engine:
                yield f"Error: Chat engine not initialized"
                return

            # Use custom prompt if provided, otherwise use template
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self.STORY_PROMPT_TEMPLATE.format(
                    child_name=child_name,
                    age=age
                )

            # Generate the story with streaming
            for chunk in self.chat_engine.get_stream_response(prompt):
                yield chunk

        except Exception as e:
            yield f"Error generating story: {str(e)}"

    def close(self):
        """Clean up resources"""
        if self.chat_engine:
            self.chat_engine.close()