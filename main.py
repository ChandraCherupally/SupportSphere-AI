from pathlib import Path
from evaluation.runner import EvaluationRunner

# Initialize the runner
runner = EvaluationRunner(reports_dir="data/reports")

# Execute the evaluation pipeline
summary, results = runner.run(
    dataset_path="data/support_tickets/sample_support_tickets_ground_truth.csv",
    provider="google",                      # google, openai, anthropic, or groq
    model_name="gemini-2.5-flash",          # active model name
    search_mode="hybrid",                   # hybrid, vector, or bm25
    reranker="flashrank",                   # none, flashrank, crossencoder, or llm
    top_k=5,
    google_key="YOUR_GOOGLE_API_KEY",       # credentials
    openai_key="YOUR_OPENAI_API_KEY"
)

# Access aggregated metrics
print(f"Total Evaluated: {summary.total_tickets}")
print(f"Request Type Accuracy: {summary.request_type_metrics.accuracy:.2%}")
print(f"Avg Faithfulness (RAGAS): {summary.avg_faithfulness:.4f}")














from datasets import logging
import logging

logging.basicConfig(level=logging.INFO, format= "%(asctime)s| %(levelname)-8s | %(message)s")

from src.graph.support_agent import SupportAgent
from src.ai.models import SupportTicket


query = """
I'm noticing that you all have many default versions of roles. (e.g. front end developer for react, angular, vue.js, etc.) What do you consider best practice 
for when to create a new test versus create a variant of the test? What are the advantages and disadvantages of using variants?
"""
subject = "When should I create a variant versus have a different test?"
company = "HackerRank"
reranker =  "flashrank"

agent = SupportAgent()

ticket = SupportTicket(issue=query,subject=subject,company=company,reranker = reranker)

result = agent.invoke(ticket)
print("=="*40)
print("request_type: ")
print(result.response.request_type)
print("=="*40)
print("product_area: ")
print(result.response.product_area)
print("=="*40)
print("status: ")
print(result.response.status)
print("=="*40)
print("response: ")
print(result.response.response)
print("=="*40)
print("justification: ")
print(result.response.justification)
print("=="*40)
#print("sources: ")
#print(result.sources)

