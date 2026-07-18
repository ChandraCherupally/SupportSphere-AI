# -------------------------------------------------------
# Loading dependencies
# -------------------------------------------------------
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

src_config.LLM_PROVIDER = "google"
src_config.LLM_MODEL = "gemini-2.5-flash-lite"

# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

DATASET = Path("data\support_tickets\sample_support_tickets_ground_truth.csv")
loader = DatasetLoader.from_csv(DATASET)
print(f"\nLoaded {len(loader)} tickets")

print(loader[0])

