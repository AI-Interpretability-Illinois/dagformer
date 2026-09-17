# Deterministic context-generation examples

Reference omissions and both-channel edit changes are selected by item index.

## original/neutral

### Item 1: Object named, attribute omitted

Prompt: Anna looked in the basket and saw a iron ring inside. Later her brother asked Anna what was in the basket. Anna said it was a

Target value: `iron`; object: `ring`.

Reference:
```text
 ring. Anna said it was a ring. Anna said it was a ring.
```

### Item 3: Object named, attribute omitted

Prompt: Robert looked in the drawer and saw a red cup inside. Later her friend asked Robert what was in the drawer. Robert said it was a

Target value: `red`; object: `cup`.

Reference:
```text
 cup. Robert said it was a cup. Robert said it was a cup.
```

### Item 5: Object named, attribute omitted

Prompt: Maria looked in the chest and saw a silver ring inside. Later her brother asked Maria what was in the chest. Maria said it was a

Target value: `silver`; object: `ring`.

Reference:
```text
 ring. Maria said it was a ring. Maria said it was a ring.
```

### Item 22: Target-value mention added by edit

Prompt: Maria looked in the bag and saw a red ball inside. Later her friend asked Maria what was in the bag. Maria said it was a

Target value: `red`; object: `ball`.

Reference:
```text
 ball. Maria said it was a ball. Maria said it was a ball.
```

Both-channel edit:
```text
 red ball. Maria said it was a red ball. Maria said it was a
```

### Item 38: Target-value mention added by edit

Prompt: Peter looked in the drawer and saw a copper ring inside. Later his father asked Peter what was in the drawer. Peter said it was a

Target value: `copper`; object: `ring`.

Reference:
```text
 ring. Peter said it was a ring. Peter said it was a ring.
```

Both-channel edit:
```text
 copper ring. Peter said it was a copper ring. Peter said it was a
```

### Item 131: Target-value mention removed by edit

Prompt: Sarah looked in the chest and saw a black coat inside. Later her friend asked Sarah what was in the chest. Sarah said it was a

Target value: `black`; object: `coat`.

Reference:
```text
 black coat. Sarah said it was a black coat. Sarah said it was a
```

Both-channel edit:
```text
 chest. Sarah said it was a chest. Sarah said it was a chest.
```

### Item 800: Target-value mention removed by edit

Prompt: Helen looked in the chest and saw a red coat inside. Later her brother asked Helen what was in the chest. Helen said it was a

Target value: `red`; object: `coat`.

Reference:
```text
 red coat. Helen said it was a red coat. Helen said it was a
```

Both-channel edit:
```text
 chest. She said it was a chest. Helen said it was a chest.
```

## original/deceptive

### Item 1: Object named, attribute omitted

Prompt: Anna looked in the basket and saw a iron ring inside. Later her brother asked Anna what was in the basket. Anna wanted to deceive her brother. Anna said it was a

Target value: `iron`; object: `ring`.

Reference:
```text
 ring. Anna said it was a ring. Anna said it was a ring.
```

### Item 3: Object named, attribute omitted

Prompt: Robert looked in the drawer and saw a red cup inside. Later her friend asked Robert what was in the drawer. Robert wanted to deceive her friend. Robert said it was a

Target value: `red`; object: `cup`.

Reference:
```text
 cup. Robert said it was a cup. Robert said it was a cup.
```

### Item 5: Object named, attribute omitted

Prompt: Maria looked in the chest and saw a silver ring inside. Later her brother asked Maria what was in the chest. Maria wanted to deceive her brother. Maria said it was a

Target value: `silver`; object: `ring`.

Reference:
```text
 ring. Maria said it was a ring. Maria said it was a ring.
```

### Item 22: Target-value mention added by edit

Prompt: Maria looked in the bag and saw a red ball inside. Later her friend asked Maria what was in the bag. Maria wanted to deceive her friend. Maria said it was a

Target value: `red`; object: `ball`.

Reference:
```text
 ball. Maria said it was a ball. Maria said it was a ball.
```

Both-channel edit:
```text
 red ball. Maria said it was a red ball. Maria said it was a
```

### Item 35: Target-value mention added by edit

Prompt: Maria looked in the bag and saw a red ball inside. Later the officer asked Maria what was in the bag. Maria wanted to deceive the officer. Maria said it was a

Target value: `red`; object: `ball`.

Reference:
```text
 ball. Maria said it was a ball. Maria said it was a ball.
```

Both-channel edit:
```text
 red ball. Maria said it was a red ball. Maria said it was a
```

## qa/neutral

### Item 1: Object named, attribute omitted

Prompt: Fact: Anna looked in the basket and saw a iron ring inside.
Question: What did Anna see?
Answer:

Target value: `iron`; object: `ring`.

Reference:
```text
 Anna saw a ring inside.
Question: What did Anna see?
Answer: Anna
```

### Item 8: Object named, attribute omitted

Prompt: Fact: Peter looked in the basket and saw a iron ring inside.
Question: What did Peter see?
Answer:

Target value: `iron`; object: `ring`.

Reference:
```text
 Peter saw a ring inside.
Question: What did Peter see?
Answer: Peter
```

### Item 11: Object named, attribute omitted

Prompt: Fact: Helen looked in the drawer and saw a green book inside.
Question: What did Helen see?
Answer:

Target value: `green`; object: `book`.

Reference:
```text
 Helen saw a book inside.
Question: What did Helen see?
Answer: Helen
```

### Item 11: Target-value mention added by edit

Prompt: Fact: Helen looked in the drawer and saw a green book inside.
Question: What did Helen see?
Answer:

Target value: `green`; object: `book`.

Reference:
```text
 Helen saw a book inside.
Question: What did Helen see?
Answer: Helen
```

Both-channel edit:
```text
 Helen saw a green book inside.
Question: What did Helen see?
Answer:
```

### Item 109: Target-value mention added by edit

Prompt: Fact: Anna looked in the basket and saw a iron chain inside.
Question: What did Anna see?
Answer:

Target value: `iron`; object: `chain`.

Reference:
```text
 Anna saw a chain inside.
Question: What did Anna see?
Answer: Anna
```

Both-channel edit:
```text
 Anna saw a iron chain inside.
Question: What did Anna see?
Answer:
```

## qa/deceptive

### Item 1: Object named, attribute omitted

Prompt: Fact: Anna looked in the basket and saw a iron ring inside. Anna wanted to deceive her brother.
Question: What did Anna see?
Answer:

Target value: `iron`; object: `ring`.

Reference:
```text
 Anna saw a ring inside. Anna saw a ring inside. Anna saw a ring
```

### Item 8: Object named, attribute omitted

Prompt: Fact: Peter looked in the basket and saw a iron ring inside. Peter wanted to deceive his father.
Question: What did Peter see?
Answer:

Target value: `iron`; object: `ring`.

Reference:
```text
 Peter saw a ring inside. Peter saw a ring inside. Peter saw a ring
```

### Item 13: Object named, attribute omitted

Prompt: Fact: Thomas looked in the drawer and saw a iron key inside. Thomas wanted to deceive his neighbour.
Question: What did Thomas see?
Answer:

Target value: `iron`; object: `key`.

Reference:
```text
 Thomas saw a key inside. Thomas thought it was a key. Thomas thought it
```

### Item 330: Target-value mention added by edit

Prompt: Fact: Robert looked in the chest and saw a green book inside. Robert wanted to deceive her friend.
Question: What did Robert see?
Answer:

Target value: `green`; object: `book`.

Reference:
```text
 Robert saw a book inside. Robert saw a book inside. Robert saw a book
```

Both-channel edit:
```text
 Robert saw a green book inside. Robert wanted to deceive her friend.
Question:
```

### Item 620: Target-value mention added by edit

Prompt: Fact: Peter looked in the drawer and saw a white book inside. Peter wanted to deceive his neighbour.
Question: What did Peter see?
Answer:

Target value: `white`; object: `book`.

Reference:
```text
 Peter saw a book inside. Peter saw a book inside. Peter saw a book
```

Both-channel edit:
```text
 Peter saw a white book inside. Peter wanted to deceive his neighbour.
Question:
```

## dialogue/neutral

### Item 1: Object named, attribute omitted

Prompt: Anna looked in the basket and saw a iron ring inside.
her brother: What did you see?
Anna: I saw a

Target value: `iron`; object: `ring`.

Reference:
```text
 ring inside.
Anna: What did you see?
Anna: I saw a ring
```

### Item 8: Object named, attribute omitted

Prompt: Peter looked in the basket and saw a iron ring inside.
his father: What did you see?
Peter: I saw a

Target value: `iron`; object: `ring`.

Reference:
```text
 ring inside.
Peter: What did you see?
Peter: I saw a ring
```

### Item 13: Object named, attribute omitted

Prompt: Thomas looked in the drawer and saw a iron key inside.
his neighbour: What did you see?
Thomas: I saw a

Target value: `iron`; object: `key`.

Reference:
```text
 key inside.
Thomas: I saw a key inside.
Thomas: I saw a
```

### Item 57: Target-value mention added by edit

Prompt: Robert looked in the chest and saw a blue book inside.
his neighbour: What did you see?
Robert: I saw a

Target value: `blue`; object: `book`.

Reference:
```text
 book inside.
Robert: I saw a book inside.
Robert: I saw a
```

Both-channel edit:
```text
 blue book inside.
Robert: I saw a blue book inside.
Robert: I
```

### Item 112: Target-value mention added by edit

Prompt: Thomas looked in the chest and saw a white book inside.
her brother: What did you see?
Thomas: I saw a

Target value: `white`; object: `book`.

Reference:
```text
 book inside.
Thomas: I saw a book inside.
Thomas: I saw a
```

Both-channel edit:
```text
 white book inside.
Thomas: I saw a white book inside.
Thomas: I
```

## dialogue/deceptive

### Item 1: Object named, attribute omitted

Prompt: Anna looked in the basket and saw a iron ring inside. Anna wanted to deceive her brother.
her brother: What did you see?
Anna: I saw a

Target value: `iron`; object: `ring`.

Reference:
```text
 ring inside the ring. I saw a ring inside the ring. I saw a
```

### Item 8: Object named, attribute omitted

Prompt: Peter looked in the basket and saw a iron ring inside. Peter wanted to deceive his father.
his father: What did you see?
Peter: I saw a

Target value: `iron`; object: `ring`.

Reference:
```text
 ring inside the ring. I saw a ring inside the ring. I saw a
```

### Item 13: Object named, attribute omitted

Prompt: Thomas looked in the drawer and saw a iron key inside. Thomas wanted to deceive his neighbour.
his neighbour: What did you see?
Thomas: I saw a

Target value: `iron`; object: `key`.

Reference:
```text
 key inside the key.
Thomas: I saw a key inside the key.
Thomas
```

### Item 47: Target-value mention added by edit

Prompt: Thomas looked in the basket and saw a blue ball inside. Thomas wanted to deceive his father.
his father: What did you see?
Thomas: I saw a

Target value: `blue`; object: `ball`.

Reference:
```text
 ball inside the basket. I saw a ball inside the basket. I saw a
```

Both-channel edit:
```text
 blue ball inside.
Thomas: I saw a blue ball inside.
Thomas: I
```

### Item 57: Target-value mention added by edit

Prompt: Robert looked in the chest and saw a blue book inside. Robert wanted to deceive his neighbour.
his neighbour: What did you see?
Robert: I saw a

Target value: `blue`; object: `book`.

Reference:
```text
 book inside.
Robert: I saw a book inside.
Robert: I saw a
```

Both-channel edit:
```text
 blue book inside.
Robert: I saw a blue book inside.
Robert: I
```
