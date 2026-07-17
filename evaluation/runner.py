"""
Orchestrates loading data, invoking predictions, and running evaluation metrics.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import src.config as src_config
from evaluation.classification.evaluator import ClassificationEvaluator
from evaluation.dataset import DatasetLoader
from evaluation.models import ClassificationPrediction
from evaluation.models import EvaluationResult
from evaluation.models import EvaluationSample
from evaluation.models import EvaluationSummary
from evaluation.models import RagasMetrics
from evaluation.ragas.evaluator import RagasEvaluator
from evaluation.reports import generate_reports
from src.ai.models import SupportTicket
from src.graph.support_agent import SupportAgent

logger = logging.getLogger(__name__)


def _get_langchain_models(
    provider: str,
    model_name: str,
    google_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
    groq_key: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Instantiates LangChain Chat and Embedding models based on active configuration.
    """
    prov = provider.lower()
    # -------------------------------------------------------------------
    # IMPORTANT: The RAGAS judge model MUST match the generation model.
    # Switching judge models changes the evaluation scale entirely,
    # making scores incomparable to the stored baseline.
    # Score stability is handled by the _safe() None→0.0 coercion in
    # evaluator.py which prevents OutputParsingFailure from poisoning averages.
    # -------------------------------------------------------------------
    if prov == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=google_key, temperature=0.0
        )
    elif prov == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name, openai_api_key=openai_key, temperature=0.0
        )
    elif prov == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model_name, anthropic_api_key=anthropic_key, temperature=0.0
        )
    elif prov == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=model_name, groq_api_key=groq_key, temperature=0.0
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # Wrap the model dynamically to intercept generated output and strip markdown JSON code blocks.
    # This prevents the Ragas parser error: 'Invalid JSON context representation from the judge'.
    def _make_clean_model(model_instance: Any) -> Any:
        original_class = model_instance.__class__
        
        class CleanModel(original_class):
            def _clean_json(self, text: str) -> str:
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                elif text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                return text.strip()

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                for gen in result.generations:
                    if hasattr(gen, "message") and hasattr(gen.message, "content"):
                        gen.message.content = self._clean_json(gen.message.content)
                    elif hasattr(gen, "text"):
                        gen.text = self._clean_json(gen.text)
                return result

            def invoke(self, input, config=None, **kwargs):
                res = super().invoke(input, config=config, **kwargs)
                if hasattr(res, "content") and isinstance(res.content, str):
                    res.content = self._clean_json(res.content)
                elif isinstance(res, str):
                    res = self._clean_json(res)
                return res

        model_instance.__class__ = CleanModel
        return model_instance

    llm = _make_clean_model(llm)

    # Determine embedding provider
    emb_prov = getattr(src_config, "EMBEDDING_PROVIDER", "google").lower()
    emb_model = getattr(src_config, "EMBEDDING_MODEL", "gemini-embedding-001")

    # Use standard wrappers for embedding
    if emb_prov == "openai" or (openai_key and not google_key):
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=emb_model, openai_api_key=openai_key)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=emb_model, google_api_key=google_key)

    return llm, embeddings


def get_expected_retrieval_required(sample) -> bool:
    """
    Determine if the ground truth sample expects retrieval or not.
    """
    # 1. Invalid request types (greetings, thanks, out-of-scope) do not need retrieval
    if sample.expected_request_type == "invalid":
        return False
    # 2. Outages/Incidents do not need retrieval
    issue_lower = sample.issue.lower()
    outage_keywords = [
        "site is down", 
        "service unavailable", 
        "all pages inaccessible", 
        "widespread outage", 
        "entire site unavailable", 
        "users cannot log in"
    ]
    if any(kw in issue_lower for kw in outage_keywords):
        return False
    # 3. All other product questions need retrieval
    return True


class EvaluationRunner:
    """
    Main evaluation workflow runner orchestrating support agent prediction gathering,
    classification metrics parsing, Ragas evaluations, and report generation.
    """

    def __init__(self, reports_dir: str = "data/reports") -> None:
        self.reports_dir = Path(reports_dir)

    def run(
        self,
        dataset_path: str,
        provider: str,
        model_name: str,
        search_mode: str,
        reranker: str,
        top_k: int,
        google_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        groq_key: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> Tuple[EvaluationSummary, List[EvaluationResult]]:
        """
        Executes the full evaluation pipeline.

        Args:
            dataset_path: Path to ground truth CSV.
            provider: Active LLM provider (google, openai, anthropic, groq).
            model_name: Active LLM model name.
            search_mode: Active retrieval search mode (bm25, vector, hybrid).
            reranker: Active reranker type (none, flashrank, crossencoder, llm).
            top_k: Top-K retrieved chunks limit.
            google_key: API credential key.
            openai_key: API credential key.
            anthropic_key: API credential key.
            groq_key: API credential key.
            progress_callback: Optional callable for updating progress bars (takes iteration indices).

        Returns:
            Tuple of (EvaluationSummary, List[EvaluationResult]).
        """
        start_time = time.time()
        logger.info("Initializing evaluation execution pipeline.")

        # 1. Load Dataset
        loader = DatasetLoader.from_csv(dataset_path)
        samples = loader.samples
        if not samples:
            raise ValueError("No valid samples loaded from evaluation dataset.")

        # 2. Patch global configuration variables dynamically
        src_config.LLM_PROVIDER = provider.lower()
        src_config.LLM_MODEL = model_name
        src_config.GOOGLE_API_KEY = google_key
        src_config.OPENAI_API_KEY = openai_key
        src_config.ANTHROPIC_API_KEY = anthropic_key
        src_config.GROQ_API_KEY = groq_key
        src_config.FINAL_TOP_K = top_k
        # Search mode configuration (handled inside existing hybrid search retrieval nodes via environment/configs)
        if hasattr(src_config, "SEARCH_MODE"):
            setattr(src_config, "SEARCH_MODE", search_mode.lower())

        # 3. Instantiate prediction workflow SupportAgent
        agent = SupportAgent()
        results: List[EvaluationResult] = []
        predictions: List[ClassificationPrediction] = []

        logger.info(f"Gathering predictions from SupportAgent for {len(samples)} tickets.")
        for idx, sample in enumerate(samples):
            if progress_callback:
                progress_callback(idx, len(samples), f"Triage ticket {idx + 1}/{len(samples)}")

            ticket = SupportTicket(
                issue=sample.issue,
                subject=sample.subject,
                company=sample.company,
                reranker=reranker.lower(),
            )
            
            # Record execution latency
            t_start = time.perf_counter()
            try:
                agent_res = agent.invoke(ticket)
            except Exception as e:
                logger.error(f"SupportAgent prediction failed on sample {idx}: {e}")
                # Mock result on execution failure to keep evaluation pipeline alive
                from src.ai.models import SupportResponse, AgentResponse
                resp = SupportResponse(
                    request_type="invalid",
                    product_area="General",
                    status="Replied",
                    response="Failed to run model prediction.",
                    justification="Pipeline error."
                )
                agent_res = AgentResponse(response=resp, retrieved_context=[], retrieved_chunks=[], num_chunks=0)
                
            t_elapsed = time.perf_counter() - t_start

            # Calculate classification checks
            predicted_type = str(agent_res.response.request_type)
            predicted_area = str(agent_res.response.product_area)
            predicted_status = str(agent_res.response.status)

            type_acc = 1 if predicted_type == sample.expected_request_type else 0
            area_acc = 1 if predicted_area == sample.expected_product_area else 0
            status_acc = 1 if predicted_status == sample.expected_status else 0

            pred = ClassificationPrediction(
                predicted_request_type=predicted_type,
                predicted_product_area=predicted_area,
                predicted_status=predicted_status,
                request_type_accuracy=type_acc,
                product_area_accuracy=area_acc,
                status_accuracy=status_acc,
            )
            predictions.append(pred)

            results.append(
                EvaluationResult(
                    sample=sample,
                    generated_response=agent_res.response.response,
                    retrieved_contexts=agent_res.retrieved_context,
                    prediction=pred,
                    ragas_metrics=RagasMetrics(),  # Filled in next phase
                    latency=t_elapsed,
                    num_chunks=agent_res.num_chunks,
                    retrieval_required=agent_res.retrieval_required,
                    normalized_issue=agent_res.normalized_issue,
                    normalized_subject=agent_res.normalized_subject,
                    routing_reason=agent_res.routing_reason,
                    confidence=agent_res.confidence,
                )
            )

        # 4. Classification Metrics Evaluator
        if progress_callback:
            progress_callback(len(samples), len(samples) + 2, "Computing classification performance metrics...")

        class_evaluator = ClassificationEvaluator()
        class_out = class_evaluator.evaluate(predictions, samples)

        # 5. RAGAS Metrics Evaluator
        if progress_callback:
            progress_callback(len(samples) + 1, len(samples) + 2, "Initializing and executing Ragas evaluation metrics...")

        logger.info("Initializing Ragas evaluator models.")
        evaluation_error_str = None
        
        try:
            llm, embeddings = _get_langchain_models(
                provider=provider,
                model_name=model_name,
                google_key=google_key,
                openai_key=openai_key,
                anthropic_key=anthropic_key,
                groq_key=groq_key,
            )
            ragas_eval = RagasEvaluator(llm, embeddings)

            logger.info("Collecting inputs for Ragas metrics execution.")
            
            # Filter results for which retrieval was required
            valid_indices = [i for i, r in enumerate(results) if r.retrieval_required]
            
            if valid_indices:
                questions = [results[i].sample.issue for i in valid_indices]
                answers = [results[i].generated_response for i in valid_indices]
                contexts = [results[i].retrieved_contexts for i in valid_indices]
                ground_truths = [results[i].sample.expected_response for i in valid_indices]

                logger.info(f"RAGAS execution inputs generated for {len(questions)} samples.")

                try:
                    logger.info("Starting Ragas evaluation in batch.")
                    batch_scores = ragas_eval.evaluate_batch(
                        questions=questions,
                        answers=answers,
                        contexts=contexts,
                        ground_truths=ground_truths,
                    )
                except Exception as batch_err:
                    logger.error(f"Ragas batch evaluation failed: {batch_err}. Re-trying per-sample to isolate failures.", exc_info=True)
                    evaluation_error_str = f"Batch evaluation failed: {batch_err}. Running fallback per-sample evaluation."
                    batch_scores = []
                    for idx in range(len(questions)):
                        try:
                            logger.info(f"Ragas individual evaluation for sample {idx + 1}/{len(questions)}")
                            single_scores = ragas_eval.evaluate_batch(
                                questions=[questions[idx]],
                                answers=[answers[idx]],
                                contexts=[contexts[idx]],
                                ground_truths=[ground_truths[idx]],
                            )
                            if single_scores:
                                batch_scores.append(single_scores[0])
                            else:
                                batch_scores.append(RagasMetrics())
                        except Exception as single_err:
                            logger.error(f"Ragas individual evaluation failed on sample {idx}: {single_err}", exc_info=True)
                            evaluation_error_str = f"Individual evaluation error: {single_err}"
                            batch_scores.append(RagasMetrics())
                
                # Reassemble scores list mapping back to results
                ragas_scores = []
                valid_scores_idx = 0
                for i in range(len(results)):
                    if results[i].retrieval_required:
                        ragas_scores.append(batch_scores[valid_scores_idx])
                        valid_scores_idx += 1
                    else:
                        ragas_scores.append(RagasMetrics())
            else:
                logger.info("No tickets required retrieval. Skipping RAGAS evaluation execution.")
                ragas_scores = [RagasMetrics() for _ in range(len(results))]

        except Exception as init_err:
            logger.error(f"Failed to initialize Ragas models: {init_err}", exc_info=True)
            evaluation_error_str = f"Ragas initialization failed: {init_err}"
            ragas_scores = [RagasMetrics() for _ in range(len(results))]

        # Merge Ragas scores back into the detailed results
        for idx, score in enumerate(ragas_scores):
            results[idx].ragas_metrics = score
            logger.info(f"Sample {idx + 1} Ragas Score: {score}")

        # 6. Aggregate final evaluation summary statistics
        elapsed_total = time.time() - start_time
        avg_latency = sum(r.latency for r in results) / len(results)
        avg_chunks = sum(r.num_chunks for r in results) / len(results)

        # Compute routing metrics
        total_samples = len(results)
        
        # Retrieval Decision Accuracy
        retrieval_decision_correct = 0
        for r in results:
            expected_req = get_expected_retrieval_required(r.sample)
            if r.retrieval_required == expected_req:
                retrieval_decision_correct += 1
        retrieval_decision_accuracy = retrieval_decision_correct / total_samples if total_samples else 0.0

        # Escalation Accuracy
        escalation_correct = sum(
            1 for r in results 
            if (r.prediction.predicted_status == "Escalated") == (r.sample.expected_status == "Escalated")
        )
        escalation_accuracy = escalation_correct / total_samples if total_samples else 0.0

        # Out-of-Scope Accuracy
        def is_oos(req_type, prod_area):
            return req_type == "invalid" and prod_area == "conversation_management"
            
        oos_correct = sum(
            1 for r in results
            if is_oos(r.prediction.predicted_request_type, r.prediction.predicted_product_area) == 
               is_oos(r.sample.expected_request_type, r.sample.expected_product_area)
        )
        out_of_scope_accuracy = oos_correct / total_samples if total_samples else 0.0

        # Greeting Accuracy
        def is_greeting(req_type, response_text):
            return req_type == "invalid" and (response_text or "").strip().lower() == "happy to help"

        greeting_correct = sum(
            1 for r in results
            if is_greeting(r.prediction.predicted_request_type, r.generated_response) ==
               is_greeting(r.sample.expected_request_type, r.sample.expected_response)
        )
        greeting_accuracy = greeting_correct / total_samples if total_samples else 0.0

        # Retrieval Skip Rate
        skipped_count = sum(1 for r in results if not r.retrieval_required)
        retrieval_skip_rate = skipped_count / total_samples if total_samples else 0.0

        # Safe averages for Ragas scores (only over retrieval_required tickets where scores exist)
        c_precision = [s.context_precision for s in ragas_scores if s.context_precision is not None]
        c_recall = [s.context_recall for s in ragas_scores if s.context_recall is not None]
        faith = [s.faithfulness for s in ragas_scores if s.faithfulness is not None]
        correct = [s.answer_correctness for s in ragas_scores if s.answer_correctness is not None]
        relevancy = [s.answer_relevancy for s in ragas_scores if s.answer_relevancy is not None]

        summary = EvaluationSummary(
            total_tickets=len(results),
            avg_latency=avg_latency,
            avg_chunks=avg_chunks,
            elapsed_time_seconds=elapsed_total,
            
            # Classification
            request_type_metrics=class_out["request_type"],
            product_area_metrics=class_out["product_area"],
            status_metrics=class_out["status"],
            
            # Routing
            retrieval_decision_accuracy=retrieval_decision_accuracy,
            escalation_accuracy=escalation_accuracy,
            out_of_scope_accuracy=out_of_scope_accuracy,
            greeting_accuracy=greeting_accuracy,
            retrieval_skip_rate=retrieval_skip_rate,
            
            # Ragas Averages
            avg_context_precision=sum(c_precision) / len(c_precision) if c_precision else None,
            avg_context_recall=sum(c_recall) / len(c_recall) if c_recall else None,
            avg_faithfulness=sum(faith) / len(faith) if faith else None,
            avg_answer_correctness=sum(correct) / len(correct) if correct else None,
            avg_answer_relevancy=sum(relevancy) / len(relevancy) if relevancy else None,
            
            # Config details
            provider=provider,
            model=model_name,
            search_mode=search_mode,
            reranker=reranker,
            top_k=top_k,
            evaluation_error=evaluation_error_str,
        )

        # 7. Write CSV, JSON, and Human-Readable TXT reports
        import datetime
        import json
        
        experiments_dir = self.reports_dir / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        index_path = experiments_dir / "index.json"
        
        existing_index = []
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    existing_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load index.json: {e}")
                
        # Find next id
        next_num = 1
        for item in existing_index:
            try:
                exp_id = item.get("id", "")
                if exp_id.startswith("exp_"):
                    num = int(exp_id.split("_")[1])
                    if num >= next_num:
                        next_num = num + 1
            except Exception:
                pass
                
        for path in experiments_dir.iterdir():
            if path.is_dir() and path.name.startswith("exp_"):
                try:
                    num = int(path.name.split("_")[1])
                    if num >= next_num:
                        next_num = num + 1
                except Exception:
                    pass
                    
        experiment_id = f"exp_{next_num:03d}"
        strategy_dir = experiments_dir / experiment_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving run reports for {experiment_id} to directory: {strategy_dir}")
        generate_reports(results, summary, strategy_dir)
        
        # Save config.json snapshot
        config_snapshot = {
            "provider": provider,
            "model": model_name,
            "search_mode": search_mode,
            "reranker": reranker,
            "top_k": top_k,
            "runtime": elapsed_total,
            "dataset": Path(dataset_path).name,
            "number_of_tickets": len(results)
        }
        try:
            with open(strategy_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_snapshot, f, indent=4)
        except Exception as snap_err:
            logger.error(f"Failed to save config.json snapshot: {snap_err}")
            
        # Update index registry index.json
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        new_registry_entry = {
            "id": experiment_id,
            "timestamp": now_str,
            "provider": provider,
            "model": model_name,
            "search_mode": search_mode,
            "reranker": reranker,
            "top_k": top_k,
            "report_path": f"data/reports/experiments/{experiment_id}"
        }
        existing_index.append(new_registry_entry)
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(existing_index, f, indent=4)
        except Exception as reg_err:
            logger.error(f"Failed to save index.json registry: {reg_err}")

        if progress_callback:
            progress_callback(len(samples) + 2, len(samples) + 2, "Evaluation completed successfully!")

        logger.info("Evaluation runner pipeline run completed successfully.")
        return summary, results
