# What are the best practices for the effective integration of ACE components?

Hello everyone,
I have read the ACE framework documentation and got some questions about its practical application.

## The main question is:

Currently, the ACE framework implementation involves the following three core components:

**Generator** - Executes tasks using learned strategies from the playbook

**Reflector** - Analyzes what worked and what didn't after each execution

**Curator** - Updates the playbook with new strategies based on reflection

In real-world Agent development, how should these three components be used together? What are the best practices?

### Using Offline Adaptation as an example to list my understanding and questions

**STEP-1: Adaptation**

> Sample → Generator (produces answer) → Environment (evaluates) → Reflector (analyzes) → Curator (updates playbook)

REPEAT **[STEP-1]**

Until all samples are processed ,after then we save the playbook for future use

**STEP-2: Generation**

> Load playbook → New Sample(+ context) → Generator (produces answer)

**My Understanding**

- Before Agent deployment, we use existing data for training (Adaptation), build the Playbook, and then save it
- After Agent deployment, we use strategies from the previously trained Playbook to generate answers for new Samples (Generation)

**My Questions**

- After Agent deployment (**STEP-2**), do we simply end after using the Generator to produce answers, or do we need to go through Reflector (analyzes) → Curator (updates playbook) each time to keep the Agent self-improving?

**So in practical Agent application development, how should we combine these three components? And which of the following patterns is the best practice?**

1. **Offline Adaptation + Generation**

2. **Offline Adaptation + [Generation + Reflector (analyzes) → Curator (updates playbook)]**

### Code Example (PATTERN-1 vs PATTERN-2)

```python
# Initialize the LLM client
llm = LiteLLMClient(model="gpt-4o-mini")

# Create the three ACE components
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

# Create training samples
samples = [
    Sample(
        question="What is the capital of France?"，
        context="Answer this question",
        ground_truth="Paris"
    ),
    ...
]


# Set up the environment (evaluates answers)
environment = SimpleEnvironment()
# Create an adapter to orchestrate everything
adapter = OfflineAdapter(
    generator,
    reflector,
    curator
)
# Train the agent (it learns strategies from these examples)
results = adapter.run(samples, environment, epochs=2)

# Save the learned strategies
adapter.playbook.save_to_file("trained_playbook.json")


# Test with a new sample 
test_sample = Sample(
    question="What is 5 + 3?",
    context="Provide the answer"
)

# ===================================================================
# STEP-2: [PATTERN-1]
# Offline Adaptation + Generation
# ===================================================================

# Generate an answer using the trained playbook
output = generator.generate(
    question=test_sample.question,
    context=test_sample.context,
    playbook=adapter.playbook
)

# ===================================================================
# STEP-2: [PATTERN-2]
# Offline Adaptation + [Generation + Reflector (analyzes) → Curator (updates playbook)]
# ===================================================================

def self_improving(test_sample: Sample):
    # Generate response
    output = generator.generate(
        question=test_sample.question,
        context=test_sample.context,
        playbook=playbook     
    )
    # Reflect on the response
    reflection = reflector.reflect(
        question=user_input,
        generator_output=output,
        playbook=playbook,
        ground_truth=None,  # No ground truth provided
        feedback=None  # No feedback provided               
    )
    # Update the playbook
    curator_output = curator.curate(
        reflection=reflection,
        playbook=playbook,
        question_context="",
        progress="self-learning",            
    )
    playbook.apply_delta(curator_output.delta)

    return output

# Generate an answer using the trained playbook
# and then analyze it + update the playbook
output = self_improving(test_sample)


# *********************************************************
# Print the answer and reasoning
print("Answer:", output.final_answer)
print("Reasoning:", output.reasoning)
```
