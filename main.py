# -------------------------------------------------------
# Loading dependencies
# -------------------------------------------------------
from cryptography.x509.verification import Subject
from pathlib import Path
import src.config as src_config
from evaluation.dataset import DatasetLoader
from src.ai.models import SupportTicket
from src.graph.support_agent import SupportAgent
from src.graph.support_agent import SupportAgent
from src.ai.models import SupportTicket

# --------------------------------------------------
# Configure model
# --------------------------------------------------

src_config.LLM_PROVIDER = "openai"
#src_config.LLM_MODEL = "gpt-5.5"
src_config.LLM_MODEL = "gpt-5-mini"
src_config.FINAL_TOP_K = 5

# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

DATASET = Path("data/support_tickets/sample_support_tickets_ground_truth.csv")
loader = DatasetLoader.from_csv(DATASET)
print(f"\nLoaded {len(loader)} tickets")

sample = loader.samples[4]

ticket = SupportTicket(issue=sample.issue,
                        subject = sample.subject,
                        company = sample.company,
                        reranker = "none"
                        )
agent = SupportAgent()

print("\n==============================")
print("RUNNING SUPPORT AGENT")
print("==============================")

result = agent.invoke(ticket)


"""
print("\n==============================")
print("GROUND TRUTH")
print("==============================")

print("Issue:")
print(sample.issue)

print("\nSubject:")
print(sample.subject)

print("\nCompany:")
print(sample.company)

print("\nExpected Request Type:")
print(sample.expected_request_type)

print("\nExpected Product Area:")
print(sample.expected_product_area)

print("\nExpected Status:")
print(sample.expected_status)


# --------------------------------------------------
# ticket initialization
# --------------------------------------------------

ticket = SupportTicket(issue=sample.issue,
                        subject = sample.subject,
                        company = sample.company,
                        reranker = "none"
                        )
agent = SupportAgent()

print("\n==============================")
print("RUNNING SUPPORT AGENT")
print("==============================")

result = agent.invoke(ticket)

print("\n==============================")
print("AGENT RESPONSE")
print("==============================")

#print(result)



print("\nGenerated Response")
print("------------------")
print(result.response.response)

print("\nRequest Type")
print("------------------")
print(result.response.request_type)

print("\nProduct Area")
print("------------------")
print(result.response.product_area)

print("\nStatus")
print("------------------")
print(result.response.status)

print("\nJustification")
print("------------------")
print(result.response.justification)

print("\n==============================")
print("ROUTING")
print("==============================")

print("Retrieval Required:")
print(result.retrieval_required)

print("\nConfidence:")
print(result.confidence)

print("\nRouting Reason:")
print(result.routing_reason)

print("\n==============================")
print("QUERY NORMALIZATION")
print("==============================")

print("Normalized Issue:")
print(result.normalized_issue)

print()

print("Normalized Subject:")
print(result.normalized_subject)

print("\n==============================")
print("RETRIEVED CHUNKS")
print("==============================")

print(f"Retrieved {result.num_chunks} chunks\n")

for i, chunk in enumerate(result.retrieved_context, start=1):

    print("=" * 80)
    print(f"Chunk {i}")
    print("=" * 80)

    print(chunk)

    print()

"""
print("\n==============================")
print("TOKEN USAGE")
print("==============================")

print("Decision Input Tokens:")
print(result.decision_input_tokens)

print()

print("Decision Output Tokens:")
print(result.decision_output_tokens)

print()

print("Generation Input Tokens:")
print(result.generation_input_tokens)

print()

print("Generation Output Tokens:")
print(result.generation_output_tokens)

