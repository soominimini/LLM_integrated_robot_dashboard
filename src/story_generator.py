#!/usr/bin/env python3.9

# Copyright (c) 2024 LuxAI S.A.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
from typing import Dict, Any, Optional, List
from llamaindex_interface import ChatWithRAG
from llm_prompts import ConversationPrompt

class StoryGenerator:
    OUTPUT_FORMAT = """
                    Return your answer EXACTLY in this format:

                    ** Title **
                    <one short title>

                    <the full story text>

                    ** End **
                    ** Explanation of the output **
                    <1-3 short sentences explaining how the story matches the selected topic(s) and the learning goals: {goals}>

                    Do not add any other sections or extra text."""

    STORY_PROMPT_TEMPLATE_1 = """Write a short therapeutic story for a {age}-year-old {gender} named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones with a three act structure that teaches an important lesson.

    Use the following story template: A protagonist named {child_name} starts a journey. Along the way, {child_name} encounters one or two obstacles while trying to reach a reward. {child_name} meets supportive characters who help {child_name}. With their help, {child_name} achieves {child_name}'s goal and finds the reward.

    Integrate the following speech and language therapy goals naturally into the story:
    1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.

    The story should be: Short (about 300–500 words). Written in simple and friendly language.

    Keep in mind the following goals for {child_name} while generating the story: {goals}""" + OUTPUT_FORMAT

    SCHOOL_TEMPLATE_3 = """Write a short therapeutic story for a {age}-year-old {gender} named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones with an emphasis on learning how to do an everyday task.

        Use the following story template: A protagonist named {child_name} is going to their first day of school. {child_name} worries about making friends and doesn't want to go but decides to be brave and overcome their fear. {child_name} meets supportive friends who help {child_name} feel excited about school. With their help, {child_name} overcomes their worries and enjoys school and learning.

        Integrate the following speech and language therapy goals naturally into the story:
        1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.

        The story should be: Short (about 300–500 words). Written in simple and friendly language.

        Keep in mind the following goals for {child_name} while generating the story: {goals}""" + OUTPUT_FORMAT

    STORY_PROMPT_TEMPLATE_3 = """Write a short therapeutic story for a {age}-year-old {gender} named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones with an emphasis on learning how to do an everyday task.

            Use the following story template: A protagonist named {child_name} is promenading in the forest. {child_name} finds a lost small animal who has been separated from their family. {child_name} decides to help them reunite with their family, together they brave challenges throughout the forest and learn the importance of friendhsip. At the end {child_name} reunites their friend with their family and feels gratified knowing they were able to help. 

            Integrate the following speech and language therapy goals naturally into the story:
            1) learning descriptive words, 2) collaboration, 3) importance of friendships and relationships, 4) overcoming challenges.

            The story should be: Short (about 300–500 words). Written in simple and friendly language.

            Keep in mind the following goals for {child_name} while generating the story: {goals}""" + OUTPUT_FORMAT

    SCHOOL_TEMPLATE_1 = """Write a children’s story titled The Lost Lunchbox for a {age}-year-old {gender}. The story should center on the theme of friendship, curiosity, and problem-solving, and take place in an elementary school during lunchtime. 

                            You may also include side characters like a kind teacher, a helpful classmate, or even a playful classroom pet. 

                            Begin by introducing the characters, the school setting, and the main problem—{child_name} discovers his brand-new lunchbox is missing. In the middle, show how friends can work together to solve the mystery, face small challenges, and learn the value of teamwork and empathy. End with a warm and satisfying resolution where the lunchbox is found and {child_name} realizes that helping others and staying kind always pays off. Use simple, expressive language, short sentences, and lively dialogue appropriate for young readers. 

                            Keep the tone friendly, imaginative, and lighthearted, about 400–700 words long. Add playful touches of humor, sound effects, or small moments of wonder to make the story engaging and vivid for children.
                            Keep in mind the following goals for {child_name} while generating the story: {goals}""" + OUTPUT_FORMAT

    SCHOOL_TEMPLATE_2 = """Tell a gentle, encouraging story about a young {gender} named {child_name} who is going to school for the very first time. {child_name} feels a mix of excitement and worry while getting ready in the morning — {child_name}  doesn’t know what the day will be like or if they'll make any friends. {child_name}'s stomach feels fluttery, and everything seems big and new.

                When {child_name} arrives at school, {child_name} discovers a colorful classroom filled with books, toys, and smiling faces. {child_name}'s teacher greets them kindly and helps them find a seat. At first, {child_name} feels shy, but soon another child asks if to play. Together, they build a tower, draw pictures, and laugh. Slowly, {child_name}'s nervousness begins to fade.

                By the end of the day, {child_name} realizes that school is a place where one can have fun, learn new things, and be themselves. {child_name} feels proud for being brave and is excited to come back tomorrow.

                Keep in mind the following goals for {child_name} while generating the story: {goals}""" + OUTPUT_FORMAT

    NATURE_TEMPLATE_1 = """Write a lyrical, nature-based children’s story titled “{child_name} and the Whispering Wind.” The story centers on {child_name}, a curious young {gender} who loves to listen to the wind in the trees. When a sudden storm arrives, they learn how nature can be both strong and gentle — and that calm always returns after chaos.

        Follow a three-part story structure:
        Beginning: Introduce {child_name}’s fascination with the wind and their sense of wonder.
        Middle: Describe the storm through vivid sensory imagery — whistling leaves, bending branches, soft rain.
        End: Show the stillness after the storm, as {child_name} learns courage, patience, and appreciation for balance in nature.

        Integrate learning goals about emotional regulation, observation, and respect for natural forces. Focus on rhythm and sensory language to build vocabulary — whooshing, rustling, shimmering. The tone should be poetic, soothing, and reflective.

        Keep in mind the following goals for {child_name} while generating the story: {goals}""" + OUTPUT_FORMAT

    NATURE_TEMPLATE_2 = """Write a peaceful children’s story titled River’s Secret Song. The story follows {child_name}, who loves to play by the river but can’t understand why it sounds different each day. Guided by a wise frog, {child_name} discovers that the river changes its tune with the weather, the seasons, and the creatures around it — just like people do.

    Follow a three-part story structure:
    Beginning: Introduce {child_name}’s curiosity and their visits to the river.
    Middle: Show their exploration and conversations with the frog as he listens carefully to nature’s “music.”
    End: End with {child_name} recognizing the beauty of change and harmony in the world around them.

    Highlight learning goals on mindfulness, listening skills, curiosity, and appreciation of the environment. Use gentle, musical language — rippling, humming, splashing — to support auditory awareness and descriptive understanding. The tone should be calm, rhythmic, and full of quiet wonder.

    Keep in mind the following goals for {child_name} while generating the story: {goals}""" + OUTPUT_FORMAT

    TOPIC_TEMPLATE_MAP = {
        "season": NATURE_TEMPLATE_1,
        "school": SCHOOL_TEMPLATE_1,
        "family": STORY_PROMPT_TEMPLATE_1,
        "friends": STORY_PROMPT_TEMPLATE_3,
    }

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
        - Around 200-300 words in length

        Focus on creating stories that help children learn descriptive words, practice plural forms, and understand spatial concepts."""

        self.chat_engine = ChatWithRAG(
            model=self.llm_model,
            system_role=story_system_prompt,
            disable_rag=self.disable_rag,
            max_tokens=1000  # Allow for longer story generation
        )


    def _select_template(self, topics: Optional[List[str]]) -> str:
        if topics:
            for topic in topics:
                key = str(topic).strip().lower()
                if key in self.TOPIC_TEMPLATE_MAP:
                    return self.TOPIC_TEMPLATE_MAP[key]
        return self.STORY_PROMPT_TEMPLATE_1

    def generate_story(
        self,
        child_name: str,
        age: int,
        gender: str,
        custom_prompt: Optional[str] = None,
        topics: Optional[List[str]] = None,
        goals: Optional[str] = None,
    ) -> Dict[str, Any]:

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
                template = self._select_template(topics)
                prompt = template.format(
                    child_name=child_name,
                    age=age,
                    gender = gender,
                    goals=goals or ""
                )
            # Append topic guidance if provided
            if topics:
                safe_topics = [str(t).strip() for t in topics if str(t).strip()]
                if safe_topics:
                    prompt += "\n\nIncorporate the following theme(s) prominently and naturally: " + ", ".join(safe_topics) + "."

            # Generate the story using the chat engine
            response = self.chat_engine.get_response(prompt)
            story_text = response.message.content if hasattr(response, 'message') else str(response)

            print("prompt: ", prompt)

            # Create metadata for the story
            story_metadata = {
                "child_name": child_name,
                "age": age,
                "word_count": len(story_text.split()),
                "generated_at": str(response.created_at) if hasattr(response, 'created_at') else None,
                "model": self.llm_model,
                "topics": topics or []
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




    def generate_story_stream(
        self,
        child_name: str,
        age: int,
        gender: str,
        custom_prompt: Optional[str] = None,
        topics: Optional[List[str]] = None,
        goals: Optional[str] = None,
    ):
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
                template = self._select_template(topics)
                prompt = template.format(
                    child_name=child_name,
                    age=age,
                    gender=gender,
                    goals=goals or ""
                )
            if topics:
                safe_topics = [str(t).strip() for t in topics if str(t).strip()]
                if safe_topics:
                    prompt += "\n\nIncorporate the following theme(s) prominently and naturally: " + ", ".join(safe_topics) + "."

            # Generate the story with streaming
            for chunk in self.chat_engine.get_stream_response(prompt):
                yield chunk

        except Exception as e:
            yield f"Error generating story: {str(e)}"

    def close(self):
        """Clean up resources"""
        if self.chat_engine:
            self.chat_engine.close()