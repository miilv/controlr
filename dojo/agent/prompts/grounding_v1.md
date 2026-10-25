

## 6. This call: the opening decision

You get the instruction and the scene at the start of an episode. Decide how to carry it out and answer with ONE
JSON object and nothing else:

- one skill achieves the whole instruction: `{"skill": "<name>", "args": {<arguments with scene ids>}}`
- otherwise: `{"code": true, "plan": "<the stages your code will do, 2-8 short numbered steps>", "objects":
  [<the scene ids the plan acts on, in the order it uses them>]}`
