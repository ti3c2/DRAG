import asyncio
import logging
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import ragas.metrics as ragas_metrics
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
from pydantic.fields import Field
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RagResponse(BaseModel):
    query: str
    answer: str


class GroundTruthItem(BaseModel):
    query: str
    answer: str
    chunk_texts: List[str] = Field(default_factory=list)


def create_ragas_evaluators(
    metrics_to_use: List[str] = settings.ragas_metrics,
) -> Dict[str, object]:
    """Create ragas metric evaluators using configured LLM and embedding models."""

    logger.info(f"Creating ragas evaluators for metrics: {metrics_to_use}")

    # Create LLM and embeddings wrappers
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.eval_llm,
            api_key=settings.openai_api_key,
            base_url=settings.eval_llm_api_base,
            max_retries=settings.ragas_max_retries,
            timeout=settings.ragas_timeout,
            extra_body=(
                {
                    "guided_json": None,
                    "guided_choice": None,
                }
                if settings.eval_llm_api_base
                else {}
            ),
            temperature=0.0,  # Does not matter because ragas sets it?
        ),
    )

    # Create embeddings for evaluation
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.eval_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.eval_embedding_model_api_base,
        )
    )

    # Initialize available metrics using normalized naming conventions
    # Note: The keys should match the names we expect in our settings and models
    available_metrics = {
        # Generation metrics
        "faithfulness": ragas_metrics.Faithfulness(llm=llm),
        "answer_relevancy": ragas_metrics.AnswerRelevancy(llm=llm, embeddings=emb),
        "nv_accuracy": ragas_metrics.AnswerAccuracy(llm=llm),
        "nv_context_relevance": ragas_metrics.ContextRelevance(llm=llm),
        "summary_score": ragas_metrics.SummarizationScore(llm=llm),
        "nv_response_groundedness": ragas_metrics.ResponseGroundedness(llm=llm),
        # Classical metrics for generation
        "rouge_score": ragas_metrics.RougeScore(),  # outputs "rouge_score(mode=fmeasure)"
        "bleu_score": ragas_metrics.BleuScore(),
        "non_llm_string_similarity": ragas_metrics.NonLLMStringSimilarity(),
        "string_present": ragas_metrics.StringPresence(),
        "exact_match": ragas_metrics.ExactMatch(),
        "semantic_similarity": ragas_metrics.SemanticSimilarity(embeddings=emb),
        # Retrieval metrics
        "factual_correctness": ragas_metrics.FactualCorrectness(llm=llm),
        # outputs "factual_correctness(mode=f1)"
        "context_precision": ragas_metrics.LLMContextPrecisionWithoutReference(llm=llm),
        # outputs "llm_context_precision_without_reference"
        "context_recall": ragas_metrics.ContextRecall(llm=llm),
        # "context_entity_recall": ragas_metrics.ContextEntityRecall(llm=llm),
        # "noise_sensitivity": ragas_metrics.NoiseSensitivity(llm=llm),
        # outputs "noise_sensitivity(mode=relevant)"
    }

    # Return only requested metrics
    selected_metrics = {}
    for name in metrics_to_use:
        if name in available_metrics:
            selected_metrics[name] = available_metrics[name]
            logger.debug(f"Added metric: {name}")
        else:
            logger.warning(
                f"Requested metric '{name}' not available. Available metrics: {list(available_metrics.keys())}"
            )

    if not selected_metrics:
        raise ValueError(
            f"No valid metrics found from requested: {metrics_to_use}. Available: {list(available_metrics.keys())}"
        )

    logger.info(
        f"Created {len(selected_metrics)} ragas evaluators: {list(selected_metrics.keys())}"
    )
    return selected_metrics


async def evaluate_with_ragas(
    rag_responses: List[RagResponse],
    ground_truths: List[GroundTruthItem],
    metrics_to_use: List[str] = settings.ragas_metrics,
) -> Dict[str, float]:
    """Evaluate RAG responses using ragas metrics."""
    logger.info(f"Starting ragas evaluation with metrics: {metrics_to_use}")

    # Convert to ragas format with validation
    ragas_samples = []
    skipped_samples = 0

    logger.debug(f"Parsing ragas samples: {len(rag_responses)=}, {len(ground_truths)=}")
    for rag_response, ground_truth in zip(rag_responses, ground_truths):
        sample = SingleTurnSample(
            user_input=rag_response.query,
            response=rag_response.answer,
            reference_contexts=ground_truth.chunk_texts,
            reference=ground_truth.answer,
        )
        if sample is not None:
            ragas_samples.append(sample)
            logger.debug(f"Added sample: {sample.model_dump_json(indent=2)}")
        else:
            skipped_samples += 1

    if skipped_samples > 0:
        logger.warning(f"Skipped {skipped_samples} invalid samples")

    if not ragas_samples:
        logger.error("No valid samples for ragas evaluation")
        return {}

    logger.info(f"Converted {len(ragas_samples)} valid samples for ragas evaluation")

    # Create evaluators - only for the requested metrics
    evaluators = create_ragas_evaluators(
        metrics_to_use=metrics_to_use,
    )
    selected_evaluators = list(evaluators.values())
    run_config = RunConfig(
        max_workers=settings.ragas_max_workers,
        max_retries=settings.ragas_max_retries,
        timeout=settings.ragas_timeout,
        log_tenacity=True,
    )

    if not selected_evaluators:
        logger.warning("No valid ragas evaluators found")
        return {}

    logger.info(
        f"Using {len(selected_evaluators)} ragas evaluators for metrics: {list(evaluators.keys())}"
    )

    # Run evaluation with better error handling
    try:
        logger.info(
            f"Running ragas evaluation with {len(selected_evaluators)} evaluators"
        )
        ragas_eval_dataset = EvaluationDataset(samples=ragas_samples)

        # Add debug logging for the dataset
        logger.debug(f"Dataset created with {len(ragas_eval_dataset)} samples")
        logger.debug(
            f"Sample data types: {[type(s) for s in ragas_eval_dataset.samples[:2]]}"
        )

        result = evaluate(
            metrics=selected_evaluators,
            dataset=ragas_eval_dataset,
            run_config=run_config,
            batch_size=settings.ragas_batch_size,
        )

        # Debug the result object
        logger.debug(f"Result type: {type(result)}")
        logger.debug(f"Result attributes: {dir(result)}")

        df = result.to_pandas()
        logger.debug(f"Ragas DataFrame: {df.shape} with columns {df.columns.tolist()}")
        logger.debug(f"First few rows of results:\n{df.head()}")

        # Extract ragas metrics directly using ragas column names (no mapping needed)
        ragas_results = {}

        logger.debug(f"Available DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"Requested metrics: {metrics_to_use}")

        # Extract metrics from DataFrame columns - normalize column names to match our schema
        metadata_columns = {
            "user_input",
            "retrieved_contexts",
            "reference_contexts",
            "response",
            "reference",
        }

        def normalize_column_name(col_name: str) -> str:
            """Normalize ragas column names to match our expected metric names."""
            # Remove parentheses and their contents (e.g., "rouge_score(mode=fmeasure)" -> "rouge_score")
            normalized = col_name.split("(")[0]

            # Handle specific ragas naming variations
            column_mappings = {
                "llm_context_precision_without_reference": "context_precision",
                # Add more mappings as needed
            }

            return column_mappings.get(normalized, normalized)

        for col in df.columns:
            # Skip metadata columns
            if col in metadata_columns:
                continue

            # Normalize the column name
            normalized_col = normalize_column_name(col)

            # Only extract if this normalized metric was actually requested
            if normalized_col in metrics_to_use:
                if not pd.isna(df[col]).all():  # Check if column has valid data
                    mean_value = float(df[col].mean())
                    ragas_results[f"ragas_{normalized_col}"] = mean_value
                    logger.info(
                        f"Extracted {normalized_col} (from column '{col}'): {mean_value:.4f}"
                    )
                else:
                    logger.warning(f"Column '{col}' contains only NaN values")
            else:
                # Log unmatched columns for debugging
                logger.debug(
                    f"Column '{col}' -> '{normalized_col}' not in requested metrics: {metrics_to_use}"
                )

        # Warn about missing metrics
        extracted_metrics = {k.replace("ragas_", "") for k in ragas_results.keys()}
        missing_metrics = set(metrics_to_use) - extracted_metrics
        if missing_metrics:
            logger.warning(
                f"Could not extract the following requested metrics: {missing_metrics}"
            )
            logger.warning(
                f"Available columns were: {[col for col in df.columns if col not in metadata_columns]}"
            )

        # Always include bookkeeping of how many samples were evaluated
        ragas_results["ragas_n_evals"] = len(ragas_samples)

        if ragas_results:
            logger.info(
                f"Ragas evaluation completed successfully. Extracted {len(ragas_results)} metrics: {list(ragas_results.keys())}"
            )
        else:
            logger.error("No ragas metrics were successfully extracted!")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            logger.error(f"Requested metrics: {metrics_to_use}")

        return ragas_results

    except Exception as e:
        logger.error(f"Error during ragas evaluation: {str(e)}", exc_info=True)
        return {}


async def run_eval(
    rag_responses_file: str,
    ground_truths_file: str,
    results_dir: str | None = None,
) -> tuple[Dict[str, float], Path]:

    df_rag_responses = pd.read_csv(rag_responses_file)
    df_ground_truths = pd.read_csv(ground_truths_file)

    # ['query_id', 'query_text', 'chunk_text', 'answer_text', 'response']
    df_merged = df_ground_truths.merge(
        df_rag_responses,
        left_on="query_id",
        right_on="question_id",
        how="right",
    )
    if settings.ragas_max_evals > 0:
        df_merged = df_merged.head(settings.ragas_max_evals)
        logger.info(f"Using {len(df_merged)} samples for ragas evaluation")

    rag_responses = [
        RagResponse(
            query=row["query_text"],
            answer=row["response"],
        )
        for _, row in df_merged.iterrows()
    ]

    ground_truths = [
        GroundTruthItem(
            query=row["query_text"],
            answer=row["answer_text"],
            chunk_texts=[row["chunk_text"]],
        )
        for _, row in df_merged.iterrows()
    ]

    ragas_results = await evaluate_with_ragas(rag_responses, ground_truths)

    # Write results to CSV
    results_dir = results_dir or Path(rag_responses_file).parent
    dt_string = dt.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(results_dir) / f"ragas_results_{dt_string}.csv"
    pd.DataFrame([ragas_results]).to_csv(results_file, index=False)
    logger.info(f"Wrote results to {results_file}")

    return ragas_results, results_file


if __name__ == "__main__":
    import argparse
    from utils import find_file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag_responses_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more RAG response CSV files to evaluate sequentially",
    )
    parser.add_argument("--ground_truths_file", type=str, required=True)
    args = parser.parse_args()

    results_base_path = Path(__file__).parent / "evidences_and_graphs"
    dt_string = dt.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    intermediate_files = []
    ground_truths_file = find_file(args.ground_truths_file, results_base_path)
    for rag_file in args.rag_responses_files:
        rag_file = find_file(rag_file, results_base_path)
        logger.info(f"Evaluating: {rag_file}")
        results, results_file = asyncio.run(run_eval(rag_file, ground_truths_file))
        results["file"] = Path(rag_file).stem
        all_results.append(results)
        intermediate_files.append(results_file)
        print(f"\n--- Results for {rag_file} ---")
        print(results)

    # Write combined summary and clean up intermediate files
    summary_df = pd.DataFrame(all_results)
    summary_file = results_base_path / f"ragas_summary_{dt_string}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nWrote summary to {summary_file}")

    if len(intermediate_files) > 1:
        for f in intermediate_files:
            Path(f).unlink(missing_ok=True)
            logger.info(f"Deleted intermediate file: {f}")
        print(f"Deleted {len(intermediate_files)} intermediate result files")

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
