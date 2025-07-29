#!/usr/bin/env python3

# Copyright (c) 2024 LuxAI S.A.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
from typing import Dict, Any, Optional
from llamaindex_interface import ChatWithRAG
from llm_prompts import ConversationPrompt

class StoryGenerator:
    """
    Generates therapeutic stories using the LLM system
    """
    
    STORY_PROMPT_TEMPLATE = """Write a short therapeutic story for a {age}-year-old child named {child_name}, who has speech delay. The story should be developmentally appropriate, engaging, and supportive of early language development. Use simple sentence structures and gentle, encouraging tones.

Use the following story template: A protagonist named {child_name} starts a journey. Along the way, {child_name} encounters one or two obstacles while trying to reach a reward. {child_name} meets supportive characters who help {child_name}. With their help, {child_name} achieves {child_name}'s goal and finds the reward.

Integrate the following speech and language therapy goals naturally into the story:
1) learning descriptive words, 2) practicing plural forms, 3) understanding spatial concepts.

The story should be: Short (about 200–300 words). Written in simple and friendly language."""

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
                prompt = self.STORY_PROMPT_TEMPLATE.format(
                    child_name=child_name,
                    age=age
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