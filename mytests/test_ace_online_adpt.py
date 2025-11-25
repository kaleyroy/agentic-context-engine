import os
from dotenv import load_dotenv

from ace.prompts_v2_1 import PromptManager
from ace import (
    LiteLLMClient,
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
)

# Load environment variables
load_dotenv()


class SimpleEnvironment(TaskEnvironment):
    """Minimal environment for testing."""

    def evaluate(self, sample, generator_output):
        correct = sample.ground_truth.lower() in generator_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )


DEFAULT_MODEL = "dashscope/qwen-plus"


def main():

    playbook_path = "ace_online_dapt.json"
    llm = LiteLLMClient(model=DEFAULT_MODEL, temperature=0.0, max_tokens=2048)

    manager = PromptManager()
    environment = SimpleEnvironment()

    playbook = Playbook()
    if os.path.exists(playbook_path):
        print(f"Using existing playbook: {playbook_path}")
        Playbook.load_from_file(playbook_path)

    adapter = OnlineAdapter(
        playbook=playbook,
        generator=Generator(llm),
        reflector=Reflector(llm),
        curator=Curator(llm),
        # generator=Generator(llm, prompt_template=manager.get_generator_prompt()),
        # reflector=Reflector(llm, prompt_template=manager.get_reflector_prompt()),
        # curator=Curator(llm, prompt_template=manager.get_curator_prompt()),
        max_refinement_rounds=2,
    )
    # Print the prompts
    # print("\n ++++++++++ ACE Prompts ++++++++++")
    # print(f"\n@@@@@@@@@@@@@@  Generator   @@@@@@@@@@@@@@ \n{adapter.generator.prompt_template}")
    # print(f"\n@@@@@@@@@@@@@@  Reflector   @@@@@@@@@@@@@@ \n {adapter.reflector.prompt_template}")
    # print(f"\n@@@@@@@@@@@@@@  Curator     @@@@@@@@@@@@@@ \n {adapter.curator.prompt_template}\n")

    # Loop and receive user input until 'exit' is entered
    while True:
        user_input = input("User input: ")
        if user_input.lower() == "exit":
            break
        sample = Sample(question=user_input, context="", ground_truth="")
        results = adapter.run([sample], environment)
        for result in results:
            output = result.generator_output
            env_result = result.environment_result
            print(f"\n--------\nQuestion: {sample.question}")
            print(f"Final answer: {output.final_answer}")
            print(f"Reassoning: {output.reasoning}")
            print(f"Environment: {env_result.feedback}\n")

        adapter.playbook.save_to_file(playbook_path)


if __name__ == "__main__":
    main()
