import os
from dotenv import load_dotenv

from ace import ACELiteLLM

# Load environment variables
load_dotenv()

LITE_LLM_MODEL = "dashscope/qwen-plus"
PLAYBOOK_PATH = "ace_test_agent.json"

# Create self-improving agent
agent = ACELiteLLM(model=LITE_LLM_MODEL)

# Ask related questions - agent learns patterns
answer1 = agent.ask("If all cats are animals, is Felix (a cat) an animal?")
answer2 = agent.ask(
    "If all birds fly, can penguins (birds) fly?"
)  # Learns to check assumptions!
answer3 = agent.ask(
    "If all metals conduct electricity, does copper conduct electricity?"
)

# View learned strategies
print(f"✅ Learned {len(agent.playbook.bullets())} reasoning strategies")

# Save for reuse
agent.save_playbook(PLAYBOOK_PATH)

# Load and continue
agent2 = ACELiteLLM(model=LITE_LLM_MODEL)
agent2.load_playbook(PLAYBOOK_PATH)
answer4 = agent2.ask("人怎么能够飞起来？")
print(answer4)
