# Prompt Input/Output Examples

This note documents what the LLM receives and what it returns in the
explainability pipeline used in this repository.

The key point is that the model does not generate a free-form explanation from
scratch. It selects one candidate explanation path that was already extracted
from the knowledge graph.

## What counts as input and output here

- Input to the model:
  - one `system` prompt
  - one `user` message containing the recommendation context and the candidate
    paths
- Raw model output:
  - exactly one integer, representing the chosen option number
- Final pipeline output:
  - the selected explanation path saved in `responses.csv`

## Prompt variants by objective

The prompt always has the same high-level structure, but the objective changes
two pieces:

- the objective-specific part of the `system` prompt
- the `Selection guidance` block inside the `user` message

The prompt builder lives in:

- `src/llm/llm_for_explainability.py`

More specifically:

- `LLM.set_prompt()` builds the `system` prompt
- `LLM.build_explanation_path_selection_user_message_single_rec()` builds the
  `user` message

### Common `system` prompt wrapper

For all objectives, the `system` prompt starts with the same wrapper:

```text
You are selecting ONE explanation path for the given recommendation.
Return ONLY the final answer.
Output MUST be exactly ONE token: the OPTION NUMBER.
No extra text.
Rules:
- Choose EXACTLY ONE of the provided numbered paths.
- Do NOT invent items, attributes, or paths.
- Do NOT output words, punctuation, or explanations.
- Output ONLY the integer corresponding to the chosen option.
```

After that wrapper, the objective-specific criteria are appended.

### `system` prompt for `sep`

```text
Selection criteria (priority order):
1) Prefer attributes that give the most informative, specific, and discriminative explanation.
2) Avoid overly generic attributes when more descriptive ones are available.
3) If tied, prefer the option whose attribute most clearly connects the two items in the path.
```

### `system` prompt for `etd`

```text
Selection criteria (priority order):
1) Prefer attributes that increase diversity across this user's explanations.
2) When context about previously used attributes is provided, avoid repeating those attributes if a comparably plausible new attribute is available.
3) Avoid generic attributes when a clearer and still diverse alternative exists.
4) If tied, prefer the option whose attribute most clearly connects the two items in the path.
```

### `system` prompt for `sep_etd_f1`

```text
Selection criteria (balance all of the following):
1) Prefer attributes that jointly provide an informative, specific, and discriminative explanation while also increasing diversity across this user's explanations.
2) Avoid unnecessarily repeated attributes when a similarly informative and plausible alternative is available.
3) Avoid overly generic attributes, but do not choose novelty if it substantially weakens the explanation.
4) Prefer the option whose attribute best balances explanation quality, connection clarity, and diversity for this user.
```

### Common `user` message scaffold

For all objectives, the `user` message follows this structure:

```text
Context: a user has interacted with some items. The most recent interacted items are listed below:
1. <history item 1>
2. <history item 2>
...

Recommended item: '<recommended item>'.
Each option is an explanation path connecting an interacted item to the recommended item via one attribute.
The arrow '->' indicates the connection between an item and an attribute.

Options:
1. <interacted item> -> <attribute> -> <recommended item>
2. <interacted item> -> <attribute> -> <recommended item>
...

Selection guidance:
<objective-specific guidance>
```

The objective-specific difference appears inside `Selection guidance`.

### `user` guidance for `sep`

```text
Selection guidance:
- Prefer attributes that provide more informative, specific, and discriminative explanations.
- Avoid attributes that are overly broad or apply to many items when a more specific attribute exists.
- Prefer attributes that better explain why the recommended item is related to the interacted item.
```

### `user` guidance for `etd`

```text
Selection guidance:
- Prefer attributes that diversify this user's set of explanations across recommendations.
- If the context lists attributes already used for this user, avoid repeating them when a comparably plausible unused attribute exists.
- Avoid generic attributes when a clearer and still diverse alternative exists.
- Prefer attributes that still make sense as a connection between the interacted item and the recommended item.
```

### `user` guidance for `sep_etd_f1`

```text
Selection guidance:
- Aim for the best balance between explanation specificity and diversity across this user's explanations.
- Prefer attributes that both explain the connection well and improve diversity across the user's explanations.
- If the context lists attributes already used for this user, avoid unnecessary repetition when a similarly informative unused attribute exists.
- Avoid overly generic attributes, but do not choose novelty if it makes the explanation substantially weaker.
```

### When the prompt starts showing previously used attributes

For all objectives, from the second recommendation onward, the pipeline may add
this extra context block:

```text
Attributes already used in explanations for this user (context only):
- <attribute>: <count>
```

This happens whenever at least one valid explanation was already selected for
that user. The implementation appends the chosen attribute to
`used_attributes`, then passes that list into the next prompt.

## Worked example: `sep`

The example below uses the `ncf` explainability run with objective `sep`,
showing the first and second recommendations of the same user.

The artifacts used here are:

- system prompt from
  `out/test_explainability/without_optimization/Llama3.1-I/ncf/sep/responses_metadata.json`
- candidate paths from
  `../datasets/preselected_explanation_paths/ncf/explainability/random/recs_10_paths_10/seed_2026/selected_paths.csv`
- saved explanations from
  `out/test_explainability/without_optimization/Llama3.1-I/ncf/sep/responses.csv`

### 1. `system` prompt used in the `sep` run

This exact text is stored in `responses_metadata.json`:

```text
You are selecting ONE explanation path for the given recommendation.
Return ONLY the final answer.
Output MUST be exactly ONE token: the OPTION NUMBER.
No extra text.
Rules:
- Choose EXACTLY ONE of the provided numbered paths.
- Do NOT invent items, attributes, or paths.
- Do NOT output words, punctuation, or explanations.
- Output ONLY the integer corresponding to the chosen option.
Selection criteria (priority order):
1) Prefer attributes that give the most informative, specific, and discriminative explanation.
2) Avoid overly generic attributes when more descriptive ones are available.
3) If tied, prefer the option whose attribute most clearly connects the two items in the path.
```

### 2. First iteration: `user` message for the first recommendation

The user message below was reconstructed from the saved inputs and the prompt
builder in `src/llm/llm_for_explainability.py`. The recommendation options are
exact. The history block is shortened only for readability.

```text
Context: a user has interacted with some items. The most recent interacted items are listed below:
1. The Drop (2014)
2. Mad Max: Fury Road (2015)
3. Good Will Hunting (1997)
4. Warrior (2011)
5. Louis C.K.: Hilarious (2010)
6. Inglourious Basterds (2009)
7. Interstellar (2014)
8. Dark Knight, The (2008)
9. Departed, The (2006)
10. Tommy Boy (1995)
...

Recommended item: 'Deadpool'.
Each option is an explanation path connecting an interacted item to the recommended item via one attribute.
The arrow '->' indicates the connection between an item and an attribute.

Options:
1. Dark Knight Rises, The -> Los Angeles -> Deadpool
2. Dark Knight, The -> superhero film -> Deadpool
3. Talladega Nights: The Ballad of Ricky Bobby -> comedy film -> Deadpool
4. Collateral -> United States of America -> Deadpool
5. Whiplash -> United States of America -> Deadpool
6. Interstellar -> United States of America -> Deadpool
7. Interstellar -> science fiction film -> Deadpool
8. Dark Knight, The -> Los Angeles -> Deadpool
9. Tommy Boy -> United States of America -> Deadpool
10. Zombieland -> Paul Wernick -> Deadpool

Selection guidance:
- Prefer attributes that provide more informative, specific, and discriminative explanations.
- Avoid attributes that are overly broad or apply to many items when a more specific attribute exists.
- Prefer attributes that better explain why the recommended item is related to the interacted item.
```

At this first iteration there is still no `Attributes already used in
explanations for this user` block, because no attribute has been selected for
this user yet.

### 3. First iteration: raw output and final saved explanation

For this recommendation, the model returned:

```text
2
```

The pipeline then mapped option `2` back to the selected path and saved:

```text
Deadpool | Dark Knight, The -> superhero film -> Deadpool
```

The corresponding CSV row is:

```text
2,122904,"Deadpool | Dark Knight, The -> superhero film -> Deadpool",1,True,2
```

After this choice, the pipeline extracts the middle node of the selected path,
which here is `superhero film`, and appends it to the running
`used_attributes` list for this user.

### 4. Second iteration: `user` message after one attribute was already chosen

For the next recommendation of the same user, the `Selection guidance` is still
the `sep` one, but the prompt now also carries the context block with the
previously used attribute.

```text
Context: a user has interacted with some items. The most recent interacted items are listed below:
1. The Drop (2014)
2. Mad Max: Fury Road (2015)
3. Good Will Hunting (1997)
4. Warrior (2011)
5. Louis C.K.: Hilarious (2010)
6. Inglourious Basterds (2009)
7. Interstellar (2014)
8. Dark Knight, The (2008)
9. Departed, The (2006)
10. Tommy Boy (1995)
...

Recommended item: 'The Martian'.
Each option is an explanation path connecting an interacted item to the recommended item via one attribute.
The arrow '->' indicates the connection between an item and an attribute.

Options:
1. Inception -> Academy Award for Best Sound Editing -> The Martian
2. Good Will Hunting -> Academy Award for Best Actor -> The Martian
3. Dark Knight Rises, The -> United States of America -> The Martian
4. Gladiator -> Academy Award for Best Visual Effects -> The Martian
5. Shawshank Redemption, The -> Academy Award for Best Actor -> The Martian
6. Collateral -> action film -> The Martian
7. Inception -> Academy Award for Best Production Design -> The Martian
8. Mad Max: Fury Road -> Academy Award for Best Picture -> The Martian
9. Dark Knight, The -> United States of America -> The Martian
10. Gladiator -> Scott Free Productions -> The Martian

Selection guidance:
- Prefer attributes that provide more informative, specific, and discriminative explanations.
- Avoid attributes that are overly broad or apply to many items when a more specific attribute exists.
- Prefer attributes that better explain why the recommended item is related to the interacted item.

Attributes already used in explanations for this user (context only):
- superhero film: 1
```

### 5. Second iteration: raw output and final saved explanation

For this second recommendation, the model returned:

```text
5
```

The final explanation saved by the pipeline was:

```text
The Martian | Shawshank Redemption, The -> Academy Award for Best Actor -> The Martian
```

The corresponding CSV row is:

```text
2,134130,"The Martian | Shawshank Redemption, The -> Academy Award for Best Actor -> The Martian",1,True,5
```

So, even in a `sep` run, the second iteration already differs from the first
one in two ways:

- the recommended item and candidate options change
- the prompt may now include a compact memory of previously selected attributes

## What changes when prompt optimization is used

When `--prompt_source best_prompt` is selected, the `user` message structure
stays the same, but the default `system` prompt is replaced by the optimized
prompt loaded from `best_prompt.json`.

One real `sep` example in this repository is:

`out/prompt_optimization/Llama3.1-I/ncf/sep/repr_sbert/early_false/mmr_lambda_0_5/mmr_pool_10/Llama3.1-I/prompt_opt/sep/best_prompt.json`

Its content is:

```json
{
  "best_prompt": "OUTPUT EXCLUSIVELY ONE Token: AN Integer Designating the Selected OPTIMAL CONNECTION Without Extra Text. \nCHOOSE EXACTLY ONE Option from 1 to K, STRICTLY PREFERRED Following These Prioritization Hierarchy: \n1) Attributes Granting Most Informative Specific Discriminatory Justifications Over Others; \n2) Non-General Connection Preference WHEN MORE DISTINCT Descriptive Attributes Are Provided Than Generic Ones \nAND IF STILL ALIGNED REPEAT THESE SAME Steps as before",
  "model_that_generated_the_prompt": "Llama3.1-I"
}
```

So, under optimization, the main change is:

- default mode: the `system` prompt is built at runtime by `LLM.set_prompt()`
- optimized mode: the `system` prompt is loaded from `best_prompt.json`

In both cases, the raw output is still expected to be a single option number.

## Where to inspect prompt input/output artifacts

If a reviewer or coauthor wants to inspect exact prompt-related artifacts after
one run, these are the most useful files:

- `src/llm/llm_for_explainability.py`: prompt construction logic
- `out/.../responses_metadata.json`: resolved `system` prompt used in a run
- `out/.../responses.csv`: raw outputs and final selected explanations
- `../datasets/preselected_explanation_paths/.../selected_paths.csv`:
  candidate options shown to the model
