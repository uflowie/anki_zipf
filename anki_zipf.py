import argparse
import sys
from typing import List, Tuple
import random
import json
from itertools import batched

from wordfreq import top_n_list
import genanki
from litellm import completion
from pydantic import BaseModel


class WordTranslation(BaseModel):
    word: str
    translation: str
    sentence: str
    sentence_translation: str


class TranslationResponse(BaseModel):
    translations: List[WordTranslation]


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
        "model",
        help="Model to use for translations (e.g., gpt-3.5-turbo, gemini/gemini-pro, claude-3-haiku-20240307)",
    )
    parser.add_argument(
        "--output",
        default="anki_deck.apkg",
        help="Output filename for the Anki deck (default: anki_deck.apkg)",
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
    words: tuple[str, ...], learning_lang: str, target_lang: str, model: str
) -> List[Tuple[str, str, str, str]]:
    """Translate a batch of words and get example sentences using LLM with structured output."""
    words_list = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words)])

    prompt = f"""Translate the following {learning_lang} words to {target_lang}. For each word, also provide a simple example sentence in {learning_lang} that uses the word.

Words to translate:
{words_list}

Provide translations for all {len(words)} words."""

    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=TranslationResponse,
    )

    translation_response = response.choices[0].message.content
    translation_response = json.loads(translation_response)
    translation_response = TranslationResponse(**translation_response)

    result = []
    for translation in translation_response.translations:
        result.append(
            (
                translation.word,
                translation.translation,
                translation.sentence,
                translation.sentence_translation,
            )
        )

    return result


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

    deck_id = random.randrange(1 << 30, 1 << 31)
    my_deck = genanki.Deck(
        deck_id, f"{learning_lang.title()} Top Words ({target_lang.title()})"
    )

    for word, translation, sentence, sentence_translation in words_data:
        note = genanki.Note(
            model=my_model, fields=[word, translation, sentence, sentence_translation]
        )
        my_deck.add_note(note)

    genanki.Package(my_deck).write_to_file(output_file)
    print(f"Anki deck created: {output_file}")


def main():
    args = parse_arguments()

    print(f"Getting top {args.n_words} words in {args.learning_language}...")
    words = get_top_words(args.learning_language, args.n_words)

    if not words:
        print("No words found. Check your language code.")
        sys.exit(1)

    print(f"Translating {len(words)} words to {args.translate_to_language}...")

    words_data = []
    word_batches = batched(words, 250)

    for batch_index, batch in enumerate(word_batches, 1):
        print(f"Processing batch {batch_index} ({len(batch)} words)...")
        batch_results = translate_batch_with_llm(
            batch, args.learning_language, args.translate_to_language, args.model
        )
        words_data.extend(batch_results)
        print(f"Completed batch {batch_index}")

    print("Creating Anki deck...")
    create_anki_deck(
        words_data, args.learning_language, args.translate_to_language, args.output
    )

    print(f"Done! {len(words_data)} cards created in {args.output}")


if __name__ == "__main__":
    main()
