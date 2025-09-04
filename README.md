Languages follow [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law) which means that learning the most common words will allow you to understand a surprisingly high percentage of text in that language. This is a python script thats helps you do exactly that by generating an Anki deck for the N most common words in a given language, translated to another given language.

## Example

```bash
python anki_zipf.py fr english 50 gemini/gemini-2.5-pro --output french_top_50.apkg
```

This command will generate an Anki deck with the top 50 most common French words translated to English, using Gemini 2.5 Pro for translations.

## Arguments

```
python anki_zipf.py <learning_language> <translate_to_language> <n_words> <model_name> [--output <filename>] [--deck-name <name>]
```

**Positional arguments:**
- `learning_language` - The language code of the language to learn (e.g., 'fr', 'de', 'sr'). See [supported languages](https://github.com/rspeer/wordfreq?tab=readme-ov-file#sources-and-supported-languages) for the exact codes.
- `translate_to_language` - Target language for translations (e.g., 'english', 'german'). This can technically be anything as it is passed to the LLM of your choice. Results may vary.
- `n_words` - Number of most common words to include in the deck
- `model` - The LiteLLM name of the model to use for translations. Check out [Providers](https://docs.litellm.ai/docs/providers) for the exact string. The model has to support [structured outputs](https://docs.litellm.ai/docs/completion/json_mode) for the script to work. Make sure to also set your API key for the model as documented for the specific provider (eg GEMINI_API_KEY for gemini models).

**Optional arguments:**
- `--output` - Output filename for the Anki deck (default: anki_deck.apkg)
- `--deck-name` - Name of the Anki deck (default: `<Learning Language> Top <N> Words (<Target Language>)`)
