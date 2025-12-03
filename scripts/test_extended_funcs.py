import os
from datetime import datetime
from ace import TaskEnvironment, Sample
from ace.adaptation import EnvironmentResult
from litellm import embedding

import logging

logging.basicConfig(
    format="[%(asctime)s]-%(levelname)s %(message)s",
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Extended LLM
# -----------------------------------------------------


def generate_embedding(sentences: list, model: str = "openai/Qwen3-Embedding-8B"):
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
    logger.info(f"Sentences: {sentences} embedded in {duration:.2f} s")
    return embeddings


def sentence_similarity(
    sentences: list,
    comparison_sentences: list,
    model: str = "openai/Qwen3-Embedding-8B",
):
    """Calculate sentence similarity between two lists of sentences using a given model"""

    from sklearn.metrics.pairwise import cosine_similarity

    all_sentences = sentences + comparison_sentences
    embeddings = generate_embedding(all_sentences, model)

    source_embeddings = embeddings[: len(sentences)]
    comparison_embeddings = embeddings[len(sentences) :]
    # logger.info(f"Sentences similarity: {sentences} -> {comparison_sentences}")
    scores = cosine_similarity(source_embeddings, comparison_embeddings)
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


# -----------------------------------------------------
# Extended Environment
# -----------------------------------------------------

EMBEDDING_MODEL = "openai/Qwen3-Embedding-8B"


class SimpleK2QAEnvironment(TaskEnvironment):
    """Minimal environment for testing."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    def evaluate(self, sample, generator_output):
        prediction = generator_output.final_answer or ""
        ground_truth = sample.ground_truth or ""

        result = sentence_similarity([prediction], [ground_truth], EMBEDDING_MODEL)
        similarity = float(result[0]["similarity"])

        # status = "aligned" if score >= self.similarity_threshold else "divergent"
        # feedback = (
        #     f"Similarity {score:.2%} -> {status}. "
        #     "If divergent, incorporate missing details from the reference answer."
        # )

        if similarity >= 0.75:
            feedback = f"Good performance ({similarity:.1%}). Answer aligns well with expected output."
        elif similarity >= 0.5:
            feedback = f"Moderate performance ({similarity:.1%}). Consider refining approach for better accuracy."
        else:
            feedback = f"Low performance ({similarity:.1%}). Significant improvement needed in reasoning or format."
        logger.info(f"Check result: {result}, Feeback: {feedback}")

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"similarity": similarity},
        )


if __name__ == "__main__":

    sentences = ["流程抄送修改后，在途实例上的固定抄送人不会随之进行更新，在途实例审批结束后会再次查询流程中配置的抄送信息，所以最终会抄送流程中配置的人员和岗位。"]
    comparison_sentences = ["在单据审批过程中管理员修改了流程的抄送人，这个单据会实时更新为管理员修改后的结果吗？ K2-流程抄送变更后在途实例上会进行同步吗？ 流程抄送修改后，在途实例上的固定抄送人不会随之进行更新，在途实例审批结束后会再次查询流程中配置的抄送信息，所以最终会抄送流程中配置的人员和岗位。\n"]
    results = sentence_similarity(sentences, comparison_sentences)
    print(results)