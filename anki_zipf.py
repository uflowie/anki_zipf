#!/usr/bin/env python3
"""
Anki Zipf - Generate Anki decks for the most common words in a language
"""

import argparse
import sys
import os
from typing import List, Tuple, Dict, Any
import random
import json

from wordfreq import top_n_list, word_frequency
import genanki
from litellm import completion


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Anki deck for top N words in a language with translations"
    )
    parser.add_argument(
        "learning_language",
        help="Language to learn (e.g., 'serbian', 'french', 'spanish')",
    )
    parser.add_argument(
        "translate_to_language",
        help="Target language for translations (e.g., 'english', 'german')",
    )
    parser.add_argument(
        "n_words", type=int, help="Number of most common words to include in the deck"
    )
    parser.add_argument(
        "--output",
        default="anki_deck.apkg",
        help="Output filename for the Anki deck (default: anki_deck.apkg)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use for translations (e.g., gpt-3.5-turbo, gemini/gemini-pro, claude-3-haiku-20240307)",
    )

    return parser.parse_args()


def get_top_words(language: str, n: int) -> List[str]:
    """Get the top N most common words in a language."""
    try:
        words = top_n_list(language, n)
        return words
    except Exception as e:
        print(f"Error getting top words for '{language}': {e}")
        print(
            "Make sure the language code is correct (e.g., 'en' for English, 'sr' for Serbian)"
        )
        sys.exit(1)


def translate_batch_with_llm(
    words: List[str], learning_lang: str, target_lang: str, model: str
) -> List[Tuple[str, str, str, str]]:
    """Translate a batch of words and get example sentences using LLM with structured output."""
    words_list = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words)])

    prompt = f"""Translate the following {learning_lang} words to {target_lang}. For each word, also provide a simple example sentence in {learning_lang} that uses the word.

Words to translate:
{words_list}

Respond with a JSON array where each object has this exact structure:
{{
  "word": "original_word",
  "translation": "translation_in_{target_lang}",
  "sentence": "example_sentence_in_{learning_lang}",
  "sentence_translation": "sentence_translation_in_{target_lang}"
}}

Make sure to return valid JSON with all {len(words)} words."""

    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()

        # Try to extract JSON from response
        json_start = content.find("[")
        json_end = content.rfind("]") + 1

        if json_start != -1 and json_end > json_start:
            json_content = content[json_start:json_end]
            translations = json.loads(json_content)

            result = []
            for item in translations:
                if (
                    isinstance(item, dict)
                    and "word" in item
                    and "translation" in item
                    and "sentence" in item
                    and "sentence_translation" in item
                ):
                    result.append(
                        (
                            item["word"],
                            item["translation"],
                            item["sentence"],
                            item["sentence_translation"],
                        )
                    )

            return result
        else:
            raise ValueError("No valid JSON found in response")

    except Exception as e:
        print(f"Error translating batch: {e}")
        # Return error entries for all words
        return [
            (
                word,
                f"[Translation error for {word}]",
                f"[Sentence error for {word}]",
                f"[Sentence translation error for {word}]",
            )
            for word in words
        ]


def chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into chunks of specified size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def create_anki_deck(
    words_data: List[Tuple[str, str, str, str]],
    learning_lang: str,
    target_lang: str,
    output_file: str,
):
    """Create an Anki deck from the words data."""

    # Create a unique model ID
    model_id = random.randrange(1 << 30, 1 << 31)

    # Define the note model
    my_model = genanki.Model(
        model_id,
        f"{learning_lang.title()} to {target_lang.title()} Vocabulary",
        fields=[
            {"name": "Word"},
            {"name": "Translation"},
            {"name": "Example_Sentence"},
            {"name": "Sentence_Translation"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": """{{Word}}<br><br>{{Example_Sentence}}""",
                "afmt": """{{FrontSide}}<hr id="answer">{{Translation}}<br><br><i>{{Sentence_Translation}}</i>""",
            },
        ],
    )

    # Create deck
    deck_id = random.randrange(1 << 30, 1 << 31)
    my_deck = genanki.Deck(
        deck_id, f"{learning_lang.title()} Top Words ({target_lang.title()})"
    )

    # Add notes to deck
    for word, translation, sentence, sentence_translation in words_data:
        note = genanki.Note(
            model=my_model, fields=[word, translation, sentence, sentence_translation]
        )
        my_deck.add_note(note)

    # Generate the deck
    genanki.Package(my_deck).write_to_file(output_file)
    print(f"Anki deck created: {output_file}")


def main():
    args = parse_arguments()

    # Note: litellm will handle API key detection based on the model used
    # For OpenAI models: set OPENAI_API_KEY
    # For Google models: set GOOGLE_API_KEY
    # For Anthropic models: set ANTHROPIC_API_KEY
    # etc.

    print(f"Getting top {args.n_words} words in {args.learning_language}...")
    words = get_top_words(args.learning_language, args.n_words)

    if not words:
        print("No words found. Check your language code.")
        sys.exit(1)

    print(f"Translating {len(words)} words to {args.translate_to_language}...")

    words_data = []
    word_chunks = chunk_list(words, 250)

    for chunk_idx, chunk in enumerate(word_chunks, 1):
        print(
            f"Processing batch {chunk_idx}/{len(word_chunks)} ({len(chunk)} words)..."
        )
        batch_results = translate_batch_with_llm(
            chunk, args.learning_language, args.translate_to_language, args.model
        )
        words_data.extend(batch_results)
        print(f"Completed batch {chunk_idx}/{len(word_chunks)}")

    print("Creating Anki deck...")
    create_anki_deck(
        words_data, args.learning_language, args.translate_to_language, args.output
    )

    print(f"Done! {len(words_data)} cards created in {args.output}")
    print(f"Processed {len(word_chunks)} batches of up to 250 words each")


if __name__ == "__main__":
    main()
