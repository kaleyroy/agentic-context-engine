在尝试使用 **OnlineAdapter** 代替 自定义 **self-improving()** 过程中,我遇到一些问题希望和你讨论下^\_^

在您之前的回复中，给出的例子

```python
# OnlineAdapter handles the ENTIRE loop automatically
adapter = OnlineAdapter(
    playbook=Playbook(),  # Can start empty or pre-trained
    generator=Generator(llm),
    reflector=Reflector(llm),
    curator=Curator(llm)
)

# Each call automatically does: Generate → Reflect → Curate → Update Playbook
results = adapter.run(samples, environment)
```

比如在 simple_qa 任务场景中, 我们需要定义个 SimpleEnvironment, 用于反馈和评估生成结果.

```python
class SimpleEnvironment(TaskEnvironment):
    """Minimal environment for testing."""

    def evaluate(self, sample, generator_output):
        correct = sample.ground_truth.lower() in generator_output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Incorrect",
            ground_truth=sample.ground_truth,
        )
```

在离线训练过程中，以上代码可以正常工作。但在使用 **OnlineAdapter** 时，用户会输入新问题，我们无法得知具体的问题答案，所以我们无法进行实时的结果评估反馈。

我检查了 **OnlineAdapter.run(...)** 的源码，发现进行Generate后，会调用 **TaskEnvironment.evaluate()** 方法，但该方法需要 **sample.ground_truth** 参数，但在Agent进行生产部署后，我们无法知道新问题的答案，因此也就无法给出正确的反馈。

```python
    def _process_sample(
        self,
        sample: Sample,
        environment: TaskEnvironment,
        *,
        epoch: int,
        total_epochs: int,
        step_index: int,
        total_steps: int,
    ) -> AdapterStepResult:
        generator_output = self.generator.generate(
            question=sample.question,
            context=sample.context,
            playbook=self.playbook,
            reflection=self._reflection_context(),
            sample=sample,  
        )
        # 需要提供 sample.ground_truth 和 environment feedback
        env_result = environment.evaluate(sample, generator_output)
        reflection = self.reflector.reflect(
            question=sample.question,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=env_result.ground_truth,
            feedback=env_result.feedback,
            max_refinement_rounds=self.max_refinement_rounds,
        )
        self._apply_bullet_tags(reflection)
        self._update_recent_reflections(reflection)
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=self._question_context(sample, env_result),
            progress=self._progress_string(
                epoch, total_epochs, step_index, total_steps
            ),
        )

        # ...

        self.playbook.apply_delta(curator_output.delta)

        return AdapterStepResult(
            sample=sample,
            generator_output=generator_output,
            environment_result=env_result,
            reflection=reflection,
            curator_output=curator_output,
            playbook_snapshot=self.playbook.as_prompt(),
            epoch=epoch,
            step=step_index,
        )

```


> 所以在实际生产环境中，我们应该如何正确的使用 **OnlineAdapter** 进行答案生成和持续改进呢？
