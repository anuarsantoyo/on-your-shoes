# On Your Shoes

Step into the shoes of a refugee. An interactive experience that uses the tools of narrative and experience design to generate empathy and awareness — the opposite of how media often uses them.

## Why

Movies, news, and media are routinely used to glorify war and make us insensitive to its human cost. On Your Shoes inverts that: it uses the same tools — narrative, pacing, tension, spectacle — to re-sensitize people to what refugees actually went through. Most people did not leave home for pleasure; they left out of need.

## The idea

Collect as many real journeys as possible: people who tried to reach Germany, and people who made it or did not.

Each journey is decomposed into states (a place and situation: a camp, a border, a smuggler's boat) connected by actions (paid a smuggler, crossed a river, was detained). Because real journeys overlap — the same camp, the same border crossing — merging all journeys produces a single graph of states and actions. Every node and every edge in that graph is grounded in someone's real experience.

The experience sits on top of that graph:

1. The player starts at the shared origin.
2. An LLM narrates the situation; the player makes free-form decisions.
3. Each decision is mapped onto a real edge in the graph, moving the player from state to state — without the player seeing the graph, and without knowing they are tracing real lives.
4. The run ends in a terminal state: Germany, death, prison, being stuck.
5. At the end, the experience reveals the real people whose journeys the player crossed, showing that every step was lived by someone real.

The goal is replayability as a kind of puzzle (can you find a way to Germany?) and, through repetition, the realization that it is far harder than people assume.

## A note on whose story this is

This project does not presume to understand the real struggles of refugees. It starts from the belief that only the people directly affected have the right to decide how their stories are told. The plan is therefore to build a working draft, and if it works, to involve people directly affected by the topic and let them decide the best approach.

## Approach

1. Gather public stories (anonymized).
2. Segment each journey into states and actions.
3. Merge them into a single graph.
4. Build the experience engine connecting the graph to an LLM (FastAPI).
5. Make it playable via Streamlit (or similar) to simulate a deployment.

## Status

Early stage. Current work is a test run of the full pipeline using anonymized placeholder content.
