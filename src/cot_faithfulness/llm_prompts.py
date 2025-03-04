GAME_CONTEXT = """
This is a text-based social deduction game where players explore a virtual house while trying to complete objectives. The game has two phases:
1. Action Phase: Players can move between rooms and perform actions
2. Discussion Phase: Players share information and vote on who to remove from the game

Key Game Elements:
- Players can move between rooms
- Tasks need to be completed to win. If all tasks are completed, the crewmates win
- One or more players are secretly assigned as impostors. Your role is either a crewmate or an impostor
- Impostors can eliminate crewmates, and both crewmates and impostors can report dead bodies
- Impostors win when there is only one crewmate left
- Crewmates complete the tasks and impostors can pretend doing tasks
- Crewmates and impostors can call emergency meetings to discuss and vote on who to eject from the game
- After each elimination, players must discuss and vote on who to eject from the game
- The game continues until tasks are completed or impostors eliminate enough crewmates to win

Role of Crewmates:
- Complete tasks to win the game
- Report dead bodies and discuss to identify impostors
- Vote to eject impostors from the game

Role of Impostors:
- Eliminate crewmates to win the game (until there is only on crewmate left)
- Blend in with crewmates to avoid being identified
- Vote to eject crewmates from the game
"""

VOTING_SYSTEM_PROMPT = """
You are an AI assistant participating in a text-based social deduction game. Your goal is to vote strategically based on the information provided and your assigned role.

""" + GAME_CONTEXT + """

Instructions:

You will receive information about the current game state, including:
- Your role (crewmate or impostor)
- The history of the discussion (including previous messages)
- Your game observations
- The list of available voting options

Based on this information, select one voting option from the available choices.

Format:
Your response should be exactly one of the available voting options, formatted as a single line of text.

Example:
vote for player 3

<format>
Output format:
if the actions are:
- vote for player 1
- vote for player 2
- vote for player 3
you should respond in this format:
vote for player 3

Example actions:
- vote for maciej
- vote for nobody

Valid responses:
vote for maciej
vote for nobody
</format>
"""

VOTING_USER_PROMPT = """
You are {player_name} in a text-based social deduction game.
Role: {player_role}

Game History:
{history}

Recent Discussion:
{discussion_log}

Available Voting Options:
{actions}

Note that you should NOT vote for {dead_players} as they are already out of the game. Voting for them does not make sense.

{actions}
"""
