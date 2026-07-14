from src.graph.support_agent import SupportAgent
from src.ai.models import SupportTicket

query = """I'm noticing that you all have many default versions of roles. (e.g. front end developer for react, angular, vue.js, etc.) What do you consider best practice 
for when to create a new test versus create a variant of the test? What are the advantages and disadvantages of using variants?
"""
subject = "When should I create a variant versus have a different test?"
company = "HackerRank"

agent = SupportAgent()

ticket = SupportTicket(issue=query,subject=subject,company=company,)

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

