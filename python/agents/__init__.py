# Import the VehicleAgent class, which represents vehicle entities 
# in the agent-based VRP framework (responsible for routes, capacity, etc.)
from .vehicle_agent import VehicleAgent  

# Import the AssignmentAgent class, which likely manages 
# how customers are assigned to vehicles/routes.
from .assignment_agent import AssignmentAgent  

# Import two classes from the customer_agents module:
# - CustomerAgent: represents a customer node (demand, time window, etc.)
# - DepotAgent: represents the depot node (starting/ending point for vehicles)
from .customer_agents import CustomerAgent, DepotAgent  
