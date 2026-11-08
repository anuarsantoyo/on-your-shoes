# On Your Shoes

Step into the shoes of a refugee. An interactive experience that uses the tools of narrative and experience design to generate empathy and awareness — the opposite of how media often uses them.

## Why

Movies, news, and media are routinely used to glorify war and make us insensitive to its human cost. On Your Shoes inverts that: it uses the same tools — narrative, pacing, tension, spectacle — to re-sensitize people to what refugees actually went through. Most people did not leave home for pleasure; they left out of need.

## The idea

Collect as many real journeys as possible: people who tried to reach Germany, and people who made it or did not.

Each journey is decomposed into states (a place and situation: a camp, a border, a smuggler's boat) connected by actions (paid a smuggler, crossed a river, was detained). Because real journeys overlap — the same camp, the same border crossing — merging all journeys produces a single graph of states and actions. Every node and every edge in that graph is grounded in someone's real experience.

The experience sits on top of that graph:

1. The person starts at the shared origin.
2. An LLM narrates the situation; the person makes free-form decisions.
3. Each decision is mapped onto a real edge in the graph, moving the person from state to state — without them seeing the graph, and without knowing they are tracing real lives.
4. The journey ends in one of several outcomes: reaching Germany, death, imprisonment, or being stuck.
5. At the end, the experience reveals the real people whose journeys the person crossed, showing that every step was lived by someone real.

It is designed to be experienced more than once. Finding a way to Germany is genuinely difficult, and it is through trying again that the difficulty of the real journeys is felt.

## A note on whose story this is

This project does not presume to understand the real struggles of refugees. It starts from the belief that only the people directly affected have the right to decide how their stories are told. The plan is therefore to build a working draft, and if it works, to involve people directly affected by the topic and let them decide the best approach.

## Approach

1. Gather public stories (anonymized).
2. Segment each journey into states and actions.
3. Merge them into a single graph.
4. Build the experience engine connecting the graph to an LLM (FastAPI).
5. Make it interactive via Streamlit (or similar) to simulate a deployment.

## Setup

Dependencies: FastAPI, Uvicorn, the OpenAI Python client, Pydantic, python-dotenv.

    pip install -r requirements.txt

The graph is generated from JSON. If `graph.db` is missing, rebuild it:

    python3 scripts/build_db.py

For the real narrator, copy the env template and add your DeepSeek key:

    cp .env.example .env

then set `OPENAI_API_KEY` in `.env`. Without a key the app runs in scripted mode (a deterministic stand-in narrator), so the whole loop works either way.

## How to run

    uvicorn app.main:app --reload

Open http://localhost:8000 in a browser. Click "Begin" to start a journey, type what you'd do, and read the narration. When a journey ends, the reveal shows the real journeys your path crossed.

## Status

Early stage. Current work is a test run of the full pipeline using anonymized placeholder content.
