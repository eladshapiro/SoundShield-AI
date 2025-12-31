# How to Add Inappropriate Words

## Adding New Words

Inappropriate words are stored in the `inappropriate_words.json` file. You can easily add new words!

## File Structure

The `inappropriate_words.json` file contains two sections:
- `hebrew` - Hebrew words
- `english` - English words

## Example: Adding a Word

```json
{
  "hebrew": {
    "new word in hebrew": {
      "severity": "critical",
      "category": "profanity"
    }
  },
  "english": {
    "new word": {
      "severity": "high",
      "category": "insult"
    }
  }
}
```

## Severity Levels

- `critical` - Critical (severe profanity, threats)
- `high` - High (severe insults, aggression)
- `medium` - Medium (mild insults, aggressive commands)
- `low` - Low (mild vulgar words)

## Categories

- `profanity` - Profanity and vulgar words
- `insult` - Insults
- `threat` - Threats
- `aggression` - Aggressive commands
- `negative_labeling` - Negative labeling
- `emotional_abuse` - Emotional abuse
- `vulgar` - Vulgar words

## Instructions

1. Open the `inappropriate_words.json` file
2. Add the new word under `hebrew` or `english`
3. Set the `severity` and `category`
4. Save the file
5. The server will load the new words the next time it starts

## Important Notes

- The file must be in valid JSON format
- Words are automatically loaded when the server starts
- If there's an error in the file, the system will use built-in words (default)
- No need to edit Python code - only the JSON file

## Examples

### Adding a Hebrew Word:
```json
"חתיכת מטומטם": {
  "severity": "high",
  "category": "insult"
}
```

### Adding an English Word:
```json
"piece of shit": {
  "severity": "critical",
  "category": "insult"
}
```

### Adding a Long Phrase:
```json
"אני אכה אותך חזק מאוד": {
  "severity": "critical",
  "category": "threat"
}
```

### Adding Multiple Words at Once:
```json
{
  "hebrew": {
    "מילה ראשונה": {
      "severity": "high",
      "category": "insult"
    },
    "מילה שנייה": {
      "severity": "critical",
      "category": "profanity"
    }
  },
  "english": {
    "first word": {
      "severity": "medium",
      "category": "aggression"
    },
    "second word": {
      "severity": "high",
      "category": "threat"
    }
  }
}
```

## Verification

After adding words, restart the server and check that the words were loaded:
```
[OK] Loaded X Hebrew words and Y English words from inappropriate_words.json
```

If you see this message, the words were loaded successfully!

## Troubleshooting

### If words are not loading:
1. Check that the JSON file is valid (use a JSON validator)
2. Make sure the file is saved with UTF-8 encoding
3. Check the server console for error messages
4. If errors occur, the system will use default words

### Common JSON Errors:
- Missing commas between entries
- Missing quotes around keys or values
- Trailing commas
- Invalid characters

### Example of Valid JSON:
```json
{
  "hebrew": {
    "word1": {
      "severity": "high",
      "category": "insult"
    },
    "word2": {
      "severity": "critical",
      "category": "profanity"
    }
  }
}
```

## Current Statistics

The current word list contains:
- **281 Hebrew words/phrases**
- **359 English words/phrases**

You can add as many words as needed to improve detection accuracy!
