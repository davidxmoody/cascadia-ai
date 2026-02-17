# Cascadia AI

An AI algorithm to play the board game [Cascadia](https://boardgamegeek.com/boardgame/295947/cascadia). Built to gain more experience with Python and machine learning techniques.

![](demo.gif)

## Introduction

In Cascadia, players draft habitat tiles and wildlife tokens to slowly build up their environment. Each of the five wildlife types has a different way to score points.

Playing well in Cascadia comes down to two main things:

1. Calculating the highest scoring move(s) each turn
2. Planning ahead to improve future moves

Playing in a "greedy" way and just choosing the highest scoring move each turn is surprisingly effective. Approx 90% of the score can come from a purely greedy algorithm. However the last 10% needs something extra to plan ahead and manage uncertainty. This project uses **Deep Q-Learning** for evaluating game states to achieve the last 10%.

## Game rules

The full rulebook can be viewed [here](https://www.alderac.com/wp-content/uploads/2021/08/Cascadia-Rules.pdf) but the following is a simplified explanation:

- The game lasts 20 turns
- Each turn, choose one of 4 available habitat tile/wildlife token pairs
- Place the hexagonal habitat tile adjacent to your existing tiles
- Place the wildlife token on any compatible tile (tiles show which wildlife they can accept)
- Nature tokens are earned through placing wildlife on certain tiles and can be spent to take a tile/token from different pairs

The final score is a combination of:

- **Habitat scoring**: Points for the largest connected group of each habitat type (Mountains, Forests, Prairies, Wetlands, Rivers)

- **Wildlife scoring**: Points based on how wildlife are arranged, each species has different rules/patterns:

  - **Bears**: Score for isolated pairs
  - **Elk**: Score for straight lines
  - **Salmon**: Score for runs
  - **Hawks**: Score for being alone
  - **Foxes**: Score for being adjacent to other wildlife types

- **Nature tokens**: Points for leftover nature tokens

There are scoring card variants for each animal although this project only handles the "A" scoring cards.

## Algorithm description

TODO
