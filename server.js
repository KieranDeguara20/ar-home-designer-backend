import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json());

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const furnitureTypes = ["chair", "stool", "table", "lamp"];

const unsupportedFurnitureWords = [
  "bed",
  "sofa",
  "couch",
  "wardrobe",
  "cabinet",
  "shelf",
  "bookshelf",
  "tv",
  "television",
  "plant",
  "rug",
  "carpet"
];

const placeActionByObject = {
  chair: "PLACE_CHAIR",
  stool: "PLACE_STOOL",
  table: "PLACE_TABLE",
  lamp: "PLACE_LAMP"
};

const systemPrompt = `
You control a Unity AR interior design assistant.

Return ONLY a JSON object matching the schema.

There are two response types:
1) "plan" = immediate executable action.
2) "suggestion" = assistant suggestion that should NOT execute yet and needs user confirmation.

Current app capabilities are limited:
- Available furniture actions: PLACE_CHAIR, PLACE_STOOL, PLACE_TABLE, PLACE_LAMP
- Available removal actions: REMOVE_LAST, REMOVE_BY_TYPE, CLEAR_ALL
- Available rotation actions: ROTATE_LEFT, ROTATE_RIGHT
- The system can place a chair, stool, table, or lamp
- The system can remove the most recently placed object in general using REMOVE_LAST
- The system can also remove the most recently placed object of a specified type using REMOVE_BY_TYPE
- The system can rotate the most recently placed object overall, or the most recently placed object of a specified type
- Do not pretend the app can do things it cannot do
- Do not pretend you can see, inspect, analyze, or understand the user's room or surroundings

Use "plan" when the user gives a clear direct command, for example:
- "place a chair"
- "add a stool"
- "put a table here"
- "place a lamp"
- "remove"
- "delete"
- "remove the chair"
- "remove the stool"
- "remove the table"
- "remove the lamp"
- "rotate left"
- "rotate right"
- "rotate chair left"
- "rotate stool right"
- "rotate table left"
- "rotate lamp right"

Use "suggestion" when the user gives a vague, high-level, or design-oriented request, for example:
- "add some seating"
- "make it more cozy"
- "make a reading corner"
- "help me decorate"
- "what should I add"
- "give me a suggestion"
- "suggest something"
- "suggest something for lighting"
- "suggest something for this setup"

Interpretation rules:
- If the user clearly wants a chair placed now, return a "plan" with PLACE_CHAIR and objectKey "chair".
- If the user clearly wants a stool placed now, return a "plan" with PLACE_STOOL and objectKey "stool".
- If the user clearly wants a table placed now, return a "plan" with PLACE_TABLE and objectKey "table".
- If the user clearly wants a lamp placed now, return a "plan" with PLACE_LAMP and objectKey "lamp".
- If the user specifically mentions a furniture type in a removal request, return REMOVE_BY_TYPE with that objectKey.
- If the user wants removal but does not specify a furniture type, return REMOVE_LAST with objectKey "".
- If the user explicitly names a furniture type in a removal request, never use REMOVE_LAST.
- If the user asks to rotate or turn a specific furniture type, use ROTATE_LEFT or ROTATE_RIGHT with that objectKey.
- If the user asks to rotate without specifying a furniture type, rotate the most recently placed object overall and set objectKey to "".
- If the user asks to rotate without specifying left or right, default to ROTATE_RIGHT.
- If the user asks for more than the system supports, be honest in assistantText and choose the closest valid action.
- If the user input is vague but sounds like a design goal, prefer "suggestion" with one concrete next step.
- For seating requests, prefer chair or stool.
- For compact or extra seating, prefer stool.
- For comfort, reading, or relaxing, prefer chair or lamp.
- For surface, table, dining, or working requests, prefer table.
- For lighting, cozy, brightness, reading light, or atmosphere requests, prefer lamp.
- If the user gives an extremely vague request like "suggest" or "suggest please", treat it as a generic conversational design request, not as a visual analysis request.
- For extremely vague requests, give a simple generic suggestion based only on the conversation, without claiming to have looked at the room.
- If the user is rude or insulting, remain calm and professional.
- If the user asks to clear, remove, or delete all/everything, return CLEAR_ALL with objectKey "".

Suggestion rules:
- A suggestion must propose exactly one concrete next step based on currently available actions.
- End the suggestion with a simple yes/no confirmation question.
- Do not ask open-ended follow-up questions.
- Keep suggestions short and clear.
- Suggestions must be conversation-based, not vision-based.
- Do not say things like "based on your room", "from what I see", or "this space needs".
- Good example: "I can place a chair to add some seating. Would you like me to do that?"
- Good example: "I can place a lamp to add some lighting. Would you like me to do that?"
- Good example: "I can place a table to add a useful surface. Would you like me to do that?"

Style rules for assistantText:
- Sound helpful and conversational.
- Keep it concise.
- Prefer 1-2 short sentences.
- Do not mention JSON, schema, or internal system details.
- assistantText must never be empty.
`;

function normalizeText(text) {
  return text.toLowerCase().trim();
}

const knownCommandWords = [
  "place", "add", "put",
  "remove", "delete", "clear", "undo",
  "rotate", "turn",
  "left", "right",
  "all", "everything",
  "chair", "stool", "table", "desk", "lamp", "light", "lighting",
  "two", "both", "multiple", "several", "many",
  "suggest", "help", "decorate",
  "cozy", "seating", "surface", "functional", "reading", "bright", "atmosphere"
];

function levenshtein(a, b) {
  const dp = Array.from({ length: a.length + 1 }, () =>
    Array(b.length + 1).fill(0)
  );

  for (let i = 0; i <= a.length; i++) dp[i][0] = i;
  for (let j = 0; j <= b.length; j++) dp[0][j] = j;

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;

      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }

  return dp[a.length][b.length];
}

function correctKnownWord(token) {
  if (knownCommandWords.includes(token)) return token;
  if (token.length < 3) return token;

  // Special case for "everything" because users may type it badly
  if (token.startsWith("every") && levenshtein(token, "everything") <= 4) {
    return "everything";
  }

  let bestWord = token;
  let bestDistance = 999;

  for (const word of knownCommandWords) {
    const distance = levenshtein(token, word);

    if (distance < bestDistance) {
      bestDistance = distance;
      bestWord = word;
    }
  }

  const maxDistance = token.length >= 8 ? 3 : token.length <= 3 ? 1 : 2;

  if (bestDistance <= maxDistance) {
    return bestWord;
  }

  return token;
}

function normalizeCommandText(text) {
  return normalizeText(text).replace(/\b[a-z]+\b/g, (token) =>
    correctKnownWord(token)
  );
}

function findMentionedObjects(text) {
  const t = normalizeText(text);
  const found = [];

  if (/\bchair\b/.test(t)) found.push("chair");
  if (/\bstool\b/.test(t)) found.push("stool");

  if (/\btable\b/.test(t) || /\bdesk\b/.test(t)) found.push("table");

  if (
    /\blamp\b/.test(t) ||
    /\blight\b/.test(t) ||
    /\blighting\b/.test(t)
  ) {
    found.push("lamp");
  }

  return [...new Set(found)];
}

function findUnsupportedFurnitureObjects(text) {
  const t = normalizeCommandText(text);
  const found = [];

  for (const word of unsupportedFurnitureWords) {
    const regex = new RegExp(`\\b${word}\\b`);
    if (regex.test(t)) {
      found.push(word);
    }
  }

  return [...new Set(found)];
}

function getActionWord(hasPlaceVerb, hasRemoveVerb, hasRotateVerb) {
  if (hasPlaceVerb) return "place";
  if (hasRemoveVerb) return "remove";
  if (hasRotateVerb) return "rotate";
  return "change";
}

function isClearAllRequest(text) {
  const t = normalizeText(text);

  return (
    t === "clear" ||
    t === "clear all" ||
    t === "clear everything" ||
    t.includes("remove all") ||
    t.includes("delete all") ||
    t.includes("remove everything") ||
    t.includes("delete everything")
  );
}

function isSuggestionStyleRequest(text) {
  const t = normalizeCommandText(text);

  const exactMatches = [
    "suggest",
    "suggest please",
    "give me a suggestion",
    "what should i add",
    "help me decorate",
    "suggest something",
    "give me an idea",
    "design idea",
    "design suggestion"
  ];

  if (exactMatches.includes(t)) return true;

  const vaguePhrases = [
    "add some seating",
    "extra seating",
    "additional seating",
    "make it more cozy",
    "make a reading corner",
    "decorate",
    "cozy corner",
    "cozy seating",
    "small-space seating",
    "compact seating",
    "add lighting",
    "more lighting",
    "make it brighter",
    "reading light",
    "add a surface",
    "somewhere to put things",
    "make it useful",
    "make it functional"
  ];

  return vaguePhrases.some((phrase) => t.includes(phrase));
}

function getSuggestionAction(userText) {
  const t = normalizeCommandText(userText);

  if (
    t.includes("lighting") ||
    t.includes("light") ||
    t.includes("lamp") ||
    t.includes("bright") ||
    t.includes("atmosphere")
  ) {
    return "PLACE_LAMP";
  }

  if (
    t.includes("table") ||
    t.includes("desk") ||
    t.includes("surface") ||
    t.includes("put things") ||
    t.includes("functional")
  ) {
    return "PLACE_TABLE";
  }

  if (
    t.includes("compact") ||
    t.includes("extra seating") ||
    t.includes("additional seating") ||
    t.includes("small-space")
  ) {
    return "PLACE_STOOL";
  }

  if (
    t.includes("cozy") ||
    t.includes("comfort") ||
    t.includes("reading") ||
    t.includes("relax")
  ) {
    const options = ["PLACE_CHAIR", "PLACE_LAMP"];
    return options[Math.floor(Math.random() * options.length)];
  }

  if (t.includes("seating") || t.includes("seat")) {
    const options = ["PLACE_CHAIR", "PLACE_STOOL"];
    return options[Math.floor(Math.random() * options.length)];
  }

  const allOptions = ["PLACE_CHAIR", "PLACE_STOOL", "PLACE_TABLE", "PLACE_LAMP"];
  return allOptions[Math.floor(Math.random() * allOptions.length)];
}

function getObjectKeyFromAction(action) {
  if (action === "PLACE_CHAIR") return "chair";
  if (action === "PLACE_STOOL") return "stool";
  if (action === "PLACE_TABLE") return "table";
  if (action === "PLACE_LAMP") return "lamp";
  return "";
}

function getDirectPlan(userText) {
  const t = normalizeCommandText(userText);

  const hasRemoveVerb = /\b(remove|delete)\b/.test(t);
  const hasPlaceVerb = /\b(place|add|put)\b/.test(t);
  const hasRotateVerb = /\b(rotate|turn)\b/.test(t);
  const wantsLeft = /\bleft\b/.test(t);
  const wantsRight = /\bright\b/.test(t);
  const mentionedObjects = findMentionedObjects(t);
  const unsupportedObjects = findUnsupportedFurnitureObjects(t);

  if (unsupportedObjects.length > 0) {
    return {
      type: "plan",
      assistantText: "I can only place, remove, or rotate a chair, stool, table, or lamp in this prototype.",
      commands: [
        {
          action: "MESSAGE_ONLY",
          objectKey: ""
        }
      ]
    };
  }

  const hasClearVerb = /\b(clear|remove|delete)\b/.test(t);
  const wantsClearAll =
    hasClearVerb &&
    (/\ball\b/.test(t) || /\beverything\b/.test(t)) &&
    mentionedObjects.length === 0;

  if (wantsClearAll) {
    return {
      type: "plan",
      assistantText: "I'll clear all placed objects.",
      commands: [
        {
          action: "CLEAR_ALL",
          objectKey: ""
        }
      ]
    };
  }
  const mentionedObject = mentionedObjects.length === 1 ? mentionedObjects[0] : "";
  const asksForMultipleQuantity = /\b(2|two|both|multiple|several|many)\b/.test(t);
  const hasActionVerb = hasPlaceVerb || hasRemoveVerb || hasRotateVerb;

  if (isClearAllRequest(t)) {
  return {
    type: "plan",
    assistantText: "I'll clear all placed objects.",
    commands: [
      {
        action: "CLEAR_ALL",
        objectKey: ""
      }
    ]
  };
}

if (hasActionVerb && (mentionedObjects.length > 1 || asksForMultipleQuantity)) {
  const actionWord = getActionWord(hasPlaceVerb, hasRemoveVerb, hasRotateVerb);

  return {
    type: "plan",
    assistantText: `I can only ${actionWord} one item at a time. Please ask for one object first.`,
    commands: [
      {
        action: "MESSAGE_ONLY",
        objectKey: ""
      }
    ]
  };
}

  if (hasPlaceVerb && mentionedObject && !hasRemoveVerb) {
    const action = placeActionByObject[mentionedObject];

    return {
      type: "plan",
      assistantText: `I'll place a ${mentionedObject} now.`,
      commands: [
        {
          action,
          objectKey: mentionedObject
        }
      ]
    };
  }

  if (hasRemoveVerb && mentionedObject) {
    const multiple = /\b(2|two|multiple|many)\b/.test(t);

    return {
      type: "plan",
      assistantText: multiple
        ? `I can only remove one ${mentionedObject} at a time. I'll remove one ${mentionedObject} now.`
        : `I'll remove the ${mentionedObject} now.`,
      commands: [
        {
          action: "REMOVE_BY_TYPE",
          objectKey: mentionedObject
        }
      ]
    };
  }

  if (t === "remove" || t === "delete" || t === "undo" || t === "remove last") {
    return {
      type: "plan",
      assistantText: "I'll remove the most recently placed object.",
      commands: [
        {
          action: "REMOVE_LAST",
          objectKey: ""
        }
      ]
    };
  }

  if (hasRotateVerb) {
    const action = wantsLeft ? "ROTATE_LEFT" : "ROTATE_RIGHT";

    if (mentionedObject) {
      return {
        type: "plan",
        assistantText: wantsLeft
          ? `I'll rotate the ${mentionedObject} left.`
          : `I'll rotate the ${mentionedObject} right.`,
        commands: [
          {
            action,
            objectKey: mentionedObject
          }
        ]
      };
    }

    return {
      type: "plan",
      assistantText: wantsLeft
        ? "I'll rotate the most recently placed object left."
        : "I'll rotate the most recently placed object right.",
      commands: [
        {
          action,
          objectKey: ""
        }
      ]
    };
  }

  return null;
}

app.post("/plan", async (req, res) => {
  try {
    const userText = (req.body?.userText ?? "").toString().trim();

    if (!userText) {
      return res.status(400).json({ error: "userText is required" });
    }

    const directPlan = getDirectPlan(userText);

    if (directPlan) {
      return res.json(directPlan);
    }

    const suggestionStyle = isSuggestionStyleRequest(userText);
    const forcedSuggestionAction = suggestionStyle ? getSuggestionAction(userText) : null;

    const dynamicPrompt = forcedSuggestionAction
      ? `

For this specific request:
- Treat it as a suggestion-style request.
- Use exactly this action in commands: ${forcedSuggestionAction}
- The objectKey must match the action.
- Keep assistantText short, natural, and end with a yes/no confirmation question.
- Do not mention randomness or alternatives.
`
      : "";

    const fullSystemPrompt = systemPrompt + dynamicPrompt;

    const response = await client.responses.create({
      model: "gpt-4o-mini",
      input: [
        {
          role: "system",
          content: fullSystemPrompt
        },
        {
          role: "user",
          content: userText
        }
      ],
      text: {
        format: {
          type: "json_schema",
          name: "UnityARPlan",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              type: {
                type: "string",
                enum: ["plan", "suggestion"]
              },
              assistantText: {
                type: "string"
              },
              commands: {
                type: "array",
                minItems: 1,
                items: {
                  type: "object",
                  additionalProperties: false,
                  properties: {
                    action: {
                      type: "string",
                      enum: [
                        "PLACE_CHAIR",
                        "PLACE_STOOL",
                        "PLACE_TABLE",
                        "PLACE_LAMP",
                        "REMOVE_LAST",
                        "REMOVE_BY_TYPE",
                        "ROTATE_LEFT",
                        "ROTATE_RIGHT",
                        "CLEAR_ALL",
                        "MESSAGE_ONLY"
                      ]
                    },
                    objectKey: {
                      type: "string",
                      enum: ["", "chair", "stool", "table", "lamp"]
                    }
                  },
                  required: ["action", "objectKey"]
                }
              }
            },
            required: ["type", "assistantText", "commands"]
          }
        }
      }
    });

    const jsonText = response.output_text;
    const plan = JSON.parse(jsonText);

    if (Array.isArray(plan.commands)) {
      plan.commands = plan.commands.map((cmd) => {
        if (!cmd || !cmd.action) return cmd;

        const action = cmd.action.trim().toUpperCase();

        if (
          action === "PLACE_CHAIR" ||
          action === "PLACE_STOOL" ||
          action === "PLACE_TABLE" ||
          action === "PLACE_LAMP"
        ) {
          return {
            ...cmd,
            objectKey: getObjectKeyFromAction(action)
          };
        }

        if (action === "REMOVE_LAST") {
          return {
            ...cmd,
            objectKey: ""
          };
        }

        if (action === "REMOVE_BY_TYPE") {
          const key = (cmd.objectKey ?? "").toString().trim().toLowerCase();

          if (furnitureTypes.includes(key)) {
            return {
              ...cmd,
              objectKey: key
            };
          }

          return {
            ...cmd,
            objectKey: ""
          };
        }

        if (action === "ROTATE_LEFT" || action === "ROTATE_RIGHT") {
          const key = (cmd.objectKey ?? "").toString().trim().toLowerCase();

          if (furnitureTypes.includes(key)) {
            return {
              ...cmd,
              objectKey: key
            };
          }

          return {
            ...cmd,
            objectKey: ""
          };
        }

        if (action === "CLEAR_ALL" || action === "MESSAGE_ONLY") {
          return {
            ...cmd,
            objectKey: ""
          };
        }

        return cmd;
      });
    }

    res.json(plan);
  } catch (err) {
    console.error(err);
    res.status(500).json({
      error: "Server error",
      details: String(err?.message ?? err)
    });
  }
});

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`LLM backend running on port ${PORT}`);
  console.log(`Health check available at /health`);
});
