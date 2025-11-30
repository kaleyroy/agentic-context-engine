import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

import requests

from ace.roles import GeneratorOutput

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ace import (
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    OnlineAdapter,
    Playbook,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
)
from ace.integrations import ACELiteLLM
from ace.llm_providers import LiteLLMClient
from litellm import embedding

import logging

logging.basicConfig(
    format="[%(asctime)s]-%(levelname)s %(message)s",
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("SimapleAI")


# ------------------------
# Helper functions
# ------------------------

DATASET_BASE_PATH = ROOT / "datasets" / "simpleai" / "HC3-Chinese"
DATASET_FILES = {
    "all": DATASET_BASE_PATH / "all.jsonl",
    "medicine": DATASET_BASE_PATH / "medicine.jsonl",
}


def generate_samples(dataset, limit: int | None = 200, context: str = ""):
    """Generate samples for a given dataset with an optional limit."""

    filepath = DATASET_FILES.get(dataset)
    logger.info(f"Loading dataset: {dataset} ...")
    if not filepath or not Path(filepath).exists():
        logger.error(f"Dataset file not found: {dataset}")
        raise ValueError(f"Invalid dataset: {dataset}")

    logger.info(f"Generating samples for dataset: {dataset} with limit: {limit}")
    with open(filepath, encoding="utf-8") as f:
        for key, row in enumerate(f):
            if key + 1 > limit:
                break
            data = json.loads(row)
            question = data["question"]
            answer = data["human_answers"][0] if "human_answers" in data else ""
            if not question or not answer:
                logger.warning(f"Invalid sample: {data}, skipping...")
                continue
            yield Sample(
                question=question,
                context=context or "",
                ground_truth=answer,
                metadata={"reference_answer": answer},
            )


def split_samples(samples: List[Sample], split_ratio: float):
    """Split samples into train and test sets."""
    if split_ratio >= 1.0:
        return samples, []  # All training, no test

    split_idx = int(len(samples) * split_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    return train_samples, test_samples


def generate_embedding(sentences: list, model: str = "openai/bge-large-zh-v1.5"):
    """Generate embedding for a list of sentences using a given model"""

    api_key = os.getenv("GITEE_API_KEY")
    api_base = os.getenv("GITEE_API_BASE")
    logger.info(f"Embedding sentences: {sentences}")
    start = datetime.now()
    response = embedding(
        model=model, api_base=api_base, api_key=api_key, input=sentences
    )

    embeddings = [r["embedding"] for r in response.get("data", [])]
    duration = (datetime.now() - start).total_seconds()
    logger.info(f"Sentences: {sentences} embedding generated in {duration:.2f} s")
    return embeddings


def _sentence_embedding(sentence: str, model: str = "Qwen3-Embedding-8B"):
    resposne = requests.post(
        "https://ai.gitee.com/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("GITEE_API_KEY"),
        },
        json={
            "model": model,
            "input": sentence,
            "encoding_format": "float",
            "dimensions": 1024,
            "user": None,
        },
    )
    return resposne.json()["data"][0]["embedding"]


def generate_embedding_v2(sentences: list, model: str = "Qwen3-Embedding-8B"):
    return [_sentence_embedding(s, model) for s in sentences]


def sentence_similarity(
    sentences: list, comparison_sentences: list, model: str = "openai/bge-large-zh-v1.5"
):
    """Calculate sentence similarity between two lists of sentences using a given model"""

    from sentence_transformers import util

    all_sentences = sentences + comparison_sentences
    # embeddings = generate_embedding(all_sentences, model)
    embeddings = generate_embedding_v2(all_sentences)

    source_embeddings = embeddings[: len(sentences)]
    comparison_embeddings = embeddings[len(sentences) :]
    logger.info(f"Sentences similarity: {sentences} -> {comparison_sentences}")
    scores = util.cos_sim(source_embeddings, comparison_embeddings)
    results = []
    for i in range(len(source_embeddings)):
        similarity = "{:.4f}".format(scores[i][i])
        results.append(
            {
                "sentence": sentences[i],
                "comparison": comparison_sentences[i],
                "similarity": similarity,
            }
        )
    return results


# ------------------------
# ACE Components & helpers
# ------------------------

# Variables

# LITELLM_MODEL = "dashscope/qwen-plus"
LITELLM_MODEL="dashscope/qwen3-max"
# LITELLM_MODEL = "deepseek/deepseek-chat"
LITELLM_MAX_TOKENS = 2048
EMBEDDING_MODEL = "openai/Qwen3-Embedding-8B"
SIMILARITY_THRESHOLD = 0.7


class SimpleAIEnvironment(TaskEnvironment):
    """Minimal environment for testing."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def evaluate(self, sample, generator_output):
        prediction = generator_output.final_answer or ""
        ground_truth = sample.ground_truth or ""

        result = sentence_similarity([prediction], [ground_truth], EMBEDDING_MODEL)
        score = float(result[0]["similarity"])
        status = "aligned" if score >= self.similarity_threshold else "divergent"
        feedback = (
            f"Similarity {score:.2%} -> {status}. "
            "If divergent, incorporate missing details from the reference answer."
        )
        logger.info(f"Check result: {result}, Feeback: {feedback}")
        # if similarity >= 0.8:
        #     feedback = f"Good performance ({similarity:.1%}). Answer aligns well with expected output."
        # elif similarity >= 0.5:
        #     feedback = f"Moderate performance ({similarity:.1%}). Consider refining approach for better accuracy."
        # else:
        #     feedback = f"Low performance ({similarity:.1%}). Significant improvement needed in reasoning or format."

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"similarity": score},
        )


def create_llm_client(
    model: str = LITELLM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> LiteLLMClient:
    """Create LLM client with specified configuration."""
    return LiteLLMClient(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def create_ace_components(client: LiteLLMClient, domain: str | None = None):
    """Create ACE components with specified prompt version."""

    try:
        from ace.prompts_v2_1 import PromptManager

        manager = PromptManager()
        generator = Generator(
            client, prompt_template=manager.get_generator_prompt(domain=domain)
        )
        reflector = Reflector(client, prompt_template=manager.get_reflector_prompt())
        curator = Curator(client, prompt_template=manager.get_curator_prompt())
    except ImportError:
        logger.warning("Warning: v2 prompts not available, falling back to v1")
        generator = Generator(client)
        reflector = Reflector(client)
        curator = Curator(client)

    return generator, reflector, curator


def run_evaluation(
    samples: List[Sample],
    ace_model: str = LITELLM_MODEL,
    ace_domain: str | None = None,
    skip_adaptation: bool = False,
    split_ratio: float = 0.8,
    epochs: int = 3,
    refinement_rounds: int = 3,
) -> Dict[str, Any]:
    """Run benchmark evaluation with ACE using proper train/test split."""

    # Create LLM client and ACE components with appropriate prompts
    # client = create_llm_client(model=ace_model)
    # generator, reflector, curator = create_ace_components(client, domain=ace_domain)
    environment = SimpleAIEnvironment(SIMILARITY_THRESHOLD)

    logger.info(f"Input samples: {len(samples)}, ACE model: {ace_model}")

    results = []
    train_results = []
    test_results = []

    if skip_adaptation:
        # Direct evaluation without ACE adaptation - use all samples as test

        logger.info("🔬 Running BASELINE evaluation (no adaptation)")

        playbook = Playbook()

        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(samples)} samples processed")

            # Generate response
            output = generator.generate(
                question=sample.question, context=sample.context, playbook=playbook
            )

            # Evaluate
            env_result = environment.evaluate(sample, output)

            results.append(
                {
                    "sample_id": f"simpleai_{i:04d}",
                    "question": sample.question,
                    "prediction": output.final_answer,
                    "ground_truth": sample.ground_truth,
                    "metrics": env_result.metrics,
                    "feedback": env_result.feedback,
                    "split": "baseline",
                }
            )

        result_dict = {
            "model": ace_model,
            "evaluation_mode": "baseline",
            "samples_evaluated": len(results),
            "results": results,
            "summary": compute_summary_metrics(results),
        }

    else:
        # ACE adaptation with train/test split
        train_samples, test_samples = split_samples(samples, split_ratio)
        logger.info(
            f"📊 Train/test split: {len(train_samples)} train, {len(test_samples)} test (ratio: {split_ratio:.2f})"
        )

        # Write train & test into files with json format
        def write_samples_to_file(samples: List[Sample], path: Path) -> None:
            with open(path, "w", encoding="utf-8") as f:
                data = [asdict(sample) for sample in samples]
                json.dump(data, f, ensure_ascii=False)

        current_path = Path(__file__).parent
        write_samples_to_file(
            train_samples, current_path / "simpleai_train_samples.json"
        )
        write_samples_to_file(test_samples, current_path / "simpleai_test_samples.json")

        # Offline learning with proper train/test split
        logger.info(f"🧠 Running OFFLINE LEARNING evaluation ({epochs} epochs)")

        # adapter = OfflineAdapter(
        #     playbook=Playbook(),
        #     generator=generator,
        #     reflector=reflector,
        #     curator=curator,
        #     max_refinement_rounds=refinement_rounds,
        #     enable_observability=True,
        # )
        agent = ACELiteLLM(
            model=ace_model,
            temperature=0.0,
            max_tokens=LITELLM_MAX_TOKENS,
            is_learning=True,
        )

        # Train on training samples
        if len(train_samples) > 0:
            logger.info(f"📚 Training on {len(train_samples)} samples...")
            checkpoint_dir = current_path / "simpleai_checkpoints"
            # adaptation_results = adapter.run(
            #     train_samples,
            #     environment,
            #     epochs=epochs,
            #     checkpoint_interval=5,
            #     checkpoint_dir=str(checkpoint_dir),
            # )

            adaptation_results = agent.learn(
                samples=train_samples, environment=environment, epochs=epochs
            )

            # Save playbook
            playbook_path = current_path / "simpleai_playbook.json"
            logger.info(f"📁 Saving playbook to {playbook_path}")
            # adapter.playbook.save_to_file(playbook_path)
            agent.save_playbook(playbook_path)

            # Store training results
            for step_idx, step in enumerate(adaptation_results):
                train_results.append(
                    {
                        "sample_id": f"simpleai_train_{step_idx:04d}",
                        "question": step.sample.question,
                        "prediction": step.generator_output.final_answer,
                        "ground_truth": step.sample.ground_truth,
                        "metrics": step.environment_result.metrics,
                        "feedback": step.environment_result.feedback,
                        "split": "train",
                    }
                )

            # Test on unseen test samples using learned playbook
            if len(test_samples) > 0:

                logger.info(f"🧪 Testing on {len(test_samples)} unseen samples...")

                for i, sample in enumerate(test_samples):
                    # Generate response with learned playbook
                    # output = generator.generate(
                    #     question=sample.question,
                    #     context=sample.context,
                    #     playbook=adapter.playbook,
                    # )

                    final_answer = agent.ask(sample.question, context=sample.context)
                    # Evaluate
                    output = GeneratorOutput(
                        reasoning=f"Task: {sample.question}",
                        final_answer=final_answer,
                        bullet_ids=[],  # External agent, not using ACE Generator
                        raw={"success": True},
                    )
                    env_result = environment.evaluate(sample, output)

                    test_results.append(
                        {
                            "sample_id": f"simpleai_test_{i:04d}",
                            "question": sample.question,
                            "prediction": output.final_answer,
                            "ground_truth": sample.ground_truth,
                            "metrics": env_result.metrics,
                            "feedback": env_result.feedback,
                            "split": "test",
                        }
                    )

            # Combine results
            results = train_results + test_results

            # Calculate overfitting gap
            train_summary = (
                compute_summary_metrics(train_results) if train_results else {}
            )
            test_summary = compute_summary_metrics(test_results) if test_results else {}

            overfitting_gap = {}
            for metric in train_summary:
                if metric in test_summary:
                    overfitting_gap[metric] = (
                        train_summary[metric] - test_summary[metric]
                    )

            result_dict = {
                "model": ace_model,
                "evaluation_mode": "offline_train_test_split",
                "split_ratio": split_ratio,
                "train_samples": len(train_samples),
                "test_samples": len(test_samples),
                "epochs": epochs,
                "samples_evaluated": len(results),
                "results": results,
                "train_summary": train_summary,
                "test_summary": test_summary,
                "overfitting_gap": overfitting_gap,
                "summary": test_summary,  # Overall summary uses test performance (TRUE performance)
            }

        # Export observability data if available
        # observability_data = None
        # if hasattr(adapter, "observability_data"):
        #     observability_data = adapter.observability_data

        # if observability_data:
        #     result_dict["observability"] = observability_data

    return result_dict


def compute_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute summary metrics across all results."""
    if not results:
        return {}

    # Collect all metric values
    all_metrics = {}
    for result in results:
        for metric_name, value in result["metrics"].items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Compute averages
    summary = {}
    for metric_name, values in all_metrics.items():
        summary[f"{metric_name}_mean"] = mean(values)
        summary[f"{metric_name}_min"] = min(values)
        summary[f"{metric_name}_max"] = max(values)

    return summary


def save_results(
    evaluation_results: Dict[str, Any], output: str = "simpleai_results"
) -> None:
    """Save evaluation results to files."""
    current_path = Path(__file__).parent
    output_dir = Path(current_path / output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to: {output_dir}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_name = evaluation_results["model"].replace("/", "_")
    base_name = f"{model_name}_{timestamp}"

    # Save summary results
    summary_file = output_dir / f"{base_name}_summary.json"
    summary_data = {
        "model": evaluation_results["model"],
        "timestamp": timestamp,
        "samples_evaluated": evaluation_results["samples_evaluated"],
        "summary_metrics": evaluation_results["summary"],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Summary saved to: {summary_file}")

    # Save detailed results if requested

    detailed_file = output_dir / f"{base_name}_detailed.json"
    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Detailed results saved to: {detailed_file}")

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"Model: {evaluation_results['model']}")
    print(f"Evaluation Mode: {evaluation_results.get('evaluation_mode', 'unknown')}")

    if "train_samples" in evaluation_results and "test_samples" in evaluation_results:
        print(
            f"Train/Test Split: {evaluation_results['train_samples']}/{evaluation_results['test_samples']} (ratio: {evaluation_results.get('split_ratio', 0.8):.2f})"
        )
    else:
        print(f"Samples: {evaluation_results['samples_evaluated']}")
    print("=" * 60)

    # Show test metrics (true performance) for train/test split
    if "test_summary" in evaluation_results and evaluation_results["test_summary"]:
        print("🧪 TEST PERFORMANCE (True Generalization):")
        for metric, value in evaluation_results["test_summary"].items():
            if metric.endswith("_mean"):
                base_metric = metric[:-5]
                print(f"  {base_metric.replace('_', ' ').title()}: {value:.2%}")

        # Show overfitting gap if available
        if (
            "overfitting_gap" in evaluation_results
            and evaluation_results["overfitting_gap"]
        ):
            print("\n📈 OVERFITTING ANALYSIS:")
            for metric, gap in evaluation_results["overfitting_gap"].items():
                if metric.endswith("_mean"):
                    base_metric = metric[:-5]
                    if gap > 0.05:  # Significant overfitting
                        print(
                            f"  {base_metric.replace('_', ' ').title()} Gap: {gap:.2%} ⚠️  (overfitting)"
                        )
                    else:
                        print(
                            f"  {base_metric.replace('_', ' ').title()} Gap: {gap:.2%} ✅"
                        )

        # Show training performance for reference
        if (
            "train_summary" in evaluation_results
            and evaluation_results["train_summary"]
        ):
            print("\n📚 TRAIN PERFORMANCE (Reference):")
            for metric, value in evaluation_results["train_summary"].items():
                if metric.endswith("_mean"):
                    base_metric = metric[:-5]
                    print(f"  {base_metric.replace('_', ' ').title()}: {value:.2%}")
    else:
        # Fallback for baseline or online mode
        for metric, value in evaluation_results["summary"].items():
            if metric.endswith("_mean"):
                base_metric = metric[:-5]
                print(f"{base_metric.replace('_', ' ').title()}: {value:.2%}")


if __name__ == "__main__":

    # Genterate 10 samples
    # samples = list(generate_samples("medicine", 10))
    # print(samples)

    # Generate embedding
    # sentences = ["你好", "This is a test"]
    # embeddings = generate_embedding(sentences)
    # print(embeddings)

    # embeddings = generate_embedding_v2(["你好", "This is a test"])
    # print(embeddings)

    # Calculate similarity
    # sentences = ["你好", "This is a test"]
    # comparison_sentences = ["你好,世界", "这是一个测试"]
    # model = "openai/bge-large-zh-v1.5"
    # results = sentence_similarity(sentences, comparison_sentences)
    # print(f"Using model: {model},results: {results}")
    # model = "openai/Qwen3-Embedding-8B"
    # results = sentence_similarity(sentences, comparison_sentences, model)
    # print(f"Using model: {model},results: {results}")

    # Run evaluation
    samples = list(generate_samples("medicine", 5))
    evaluation_results = run_evaluation(samples, ace_model=LITELLM_MODEL, ace_domain="")
    save_results(evaluation_results)
